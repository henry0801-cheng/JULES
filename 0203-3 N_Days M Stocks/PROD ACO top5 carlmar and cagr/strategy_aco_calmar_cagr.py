import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
from copy import deepcopy

# 忽略警告訊息
warnings.simplefilter(action='ignore', category=FutureWarning)

# 設定繁體中文字型 (避免圖表亂碼)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

class Strategy:
    def __init__(self, data_file, n_days, top_k, v_bar, trail_stop, pct_stp, data=None):
        """
        初始化 N 日動能選股策略 (嚴格模式)
        
        參數:
        :param data_file: 股價資料檔路徑 (Excel)
        :param n_days: 動能回溯天數 (過去 N 日漲幅排名)
        :param top_k: 最大持倉檔數 (Target Position Count)
        :param v_bar: 流動性門檻 (單位: 千萬元，例如 9.0 代表 9000萬)
        :param trail_stop: 移動停損百分比 (例如 0.1 代表 10%)
        :param pct_stp: 固定停損百分比 (例如 0.09 代表 9%)
        :param data: (可選) 預先載入的資料字典 {'P': df, 'Q': df}
        """
        self.data_file = data_file
        self.n_days = int(n_days)
        self.top_k = int(top_k)
        self.v_bar = v_bar
        self.trail_stop = trail_stop
        self.pct_stp = pct_stp
        
        # 固定交易參數
        self.initial_capital = 20_000_000  # 初始本金 2000 萬
        self.tax_rate = 0.001              # 交易稅 0.1%
        self.slip_rate = 0.003             # 滑價 0.3% (買賣皆扣)
        
        # 載入資料
        if data is not None:
             self.data = data
        else:
             self.data = self.load_data()
        
    def load_data(self):
        """讀取並對齊股價與成交量資料"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"找不到檔案: {self.data_file}")
            
        xls = pd.ExcelFile(self.data_file)
        
        # 讀取 'P' (收盤價) 和 'Q' (成交量)
        df_p = pd.read_excel(xls, 'P', index_col=0, parse_dates=True)
        df_q = pd.read_excel(xls, 'Q', index_col=0, parse_dates=True)
        
        # 確保索引與欄位對齊
        common_cols = df_p.columns.intersection(df_q.columns)
        common_index = df_p.index.intersection(df_q.index)
        
        df_p = df_p.loc[common_index, common_cols]
        df_q = df_q.loc[common_index, common_cols]
        
        return {'P': df_p, 'Q': df_q}

    def run_backtest(self):
        """執行回測主邏輯"""
        P = self.data['P']
        Q = self.data['Q']
        dates = P.index
        
        # --- 1. 預先計算特徵 (Features) ---
        # 過去 N 日漲幅: (P_t / P_{t-N}) - 1
        returns_n = P.pct_change(self.n_days)
        
        # 成交金額 (元) = P * Q * 1000
        # 門檻轉換: V_Bar (千萬) -> 元
        turnover_value = P * Q * 1000
        v_bar_threshold = self.v_bar * 10_000_000
        
        # --- 2. 初始化回測變數 ---
        cash = self.initial_capital
        holdings = {} 
        # holdings 結構: {ticker: {'shares': 股數, 'entry_price': 進場價, 'highest_price': 最高價, 'exit_reason': 原因}}
        
        equity_curve = []   # 每日權益曲線
        trades = []         # 交易紀錄
        daily_records = []  # 每日持倉統計
        daily_candidates = [] # 每日候選名單紀錄
        equity_hold = []    # 每日持股明細
        
        rebalance_counter = 0
        self.pending_sells = [] # 待賣出清單 (T日產生，T+1日執行)
        self.pending_buys = []  # 待買入清單 (T日產生，T+1日執行)
        
        # --- 3. 逐日回測迴圈 ---
        for i in range(len(dates)):
            today = dates[i]
            
            # 前 N 天無資料，跳過但記錄資金
            if i < self.n_days:
                equity_curve.append({'Date': today, 'Equity': cash})
                equity_hold.append({'Date': today, 'Count': 0, 'Details': str({})})
                continue
            
            # 當日價格數據
            todays_prices = P.iloc[i]
            # 昨日價格 (用於計算當日漲跌幅，檢查 Limit Up/Down)
            yesterday_prices = P.iloc[i-1] if i > 0 else None
            
            # Step A: 計算當前權益 (Mark to Market)
            current_equity = cash
            current_holdings_info = {}
            for ticker, info in list(holdings.items()):
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                     current_price = todays_prices[ticker]
                     market_value = info['shares'] * current_price
                     current_equity += market_value
                     current_holdings_info[ticker] = info['shares']
            
            equity_curve.append({'Date': today, 'Equity': current_equity})
            equity_hold.append({'Date': today, 'Count': len(holdings), 'Details': str(current_holdings_info)})

            # Step B: 執行「待賣出」訂單 (Pending Sells)
            money_returned = 0
            for ticker in self.pending_sells:
                if ticker in holdings and ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    
                    # 漲跌停保護: 若當日跌幅 > 9.6%，視為跌停鎖死，無法賣出
                    if yesterday_prices is not None and ticker in yesterday_prices:
                        prev_close = yesterday_prices[ticker]
                        if prev_close > 0 and abs((price - prev_close) / prev_close) > 0.096:
                            continue # 跳過賣出，明日再試
                    
                    # 執行賣出
                    shares = holdings[ticker]['shares']
                    revenue = shares * price * (1 - self.slip_rate - self.tax_rate)
                    money_returned += revenue
                    
                    # 記錄交易
                    trades.append({
                        'Ticker': ticker,
                        'BuyDate': holdings[ticker]['entry_date'],
                        'SellDate': today,
                        'BuyPrice': holdings[ticker]['entry_price'],
                        'SellPrice': price,
                        'Shares': shares,
                        'PnL': revenue - (shares * holdings[ticker]['entry_price'] * (1 + self.slip_rate)),
                        'Reason': holdings[ticker].get('exit_reason', 'Rebalance')
                    })
                    del holdings[ticker]
            
            cash += money_returned
            self.pending_sells = [] # 清空已處理清單

            # Step C: 執行「待買入」訂單 (Pending Buys)
            # 嚴格模式: 僅買入 Top K 清單，若買不到則空手 (不遞補)
            
            for order in self.pending_buys:
                # 檢查持倉上限
                if len(holdings) >= self.top_k:
                    break
                
                ticker = order['ticker']
                budget = order['budget']
                
                if ticker in holdings: continue
                
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    
                    # 漲跌停保護: 若當日漲幅 > 9.6%，視為漲停鎖死，無法買入
                    if yesterday_prices is not None and ticker in yesterday_prices:
                        prev_close = yesterday_prices[ticker]
                        if prev_close > 0 and abs((price - prev_close) / prev_close) > 0.096:
                            continue # 放棄此檔，且不遞補 (Strict Mode)
                    
                    cost_per_share = price * (1 + self.slip_rate)
                    
                    # 資金檢查 (含部分成交邏輯)
                    if cash >= cost_per_share * 1000: # 至少買一張
                        target_shares = int(budget / cost_per_share)
                        max_shares_by_cash = int(cash / cost_per_share)
                        
                        # 部分成交: 兩者取小。若本金不足預算，則用剩餘本金買滿。
                        shares_to_buy = min(target_shares, max_shares_by_cash)
                        
                        if shares_to_buy > 0:
                            actual_cost = shares_to_buy * cost_per_share
                            cash -= actual_cost
                            holdings[ticker] = {
                                'shares': shares_to_buy,
                                'entry_price': price,
                                'entry_date': today,
                                'highest_price': price,
                                'exit_reason': ''
                            }
            self.pending_buys = [] # 清空

            # Step D: 更新最高價 & 檢查出場訊號 (Trailing/Fixed Stop)
            # 這些訊號將在「下一個交易日」執行
            next_sells = []
            
            for ticker in list(holdings.keys()):
                if ticker not in todays_prices: continue
                price = todays_prices[ticker]
                
                # 更新最高價
                if price > holdings[ticker]['highest_price']:
                    holdings[ticker]['highest_price'] = price
                
                # 檢查移動停損 (Trail Stop)
                if price < holdings[ticker]['highest_price'] * (1 - self.trail_stop):
                    holdings[ticker]['exit_reason'] = 'TrailStop'
                    if ticker not in next_sells: next_sells.append(ticker)
                
                # 檢查固定停損 (Stop Loss)
                elif price < holdings[ticker]['entry_price'] * (1 - self.pct_stp):
                    holdings[ticker]['exit_reason'] = 'StopLoss'
                    if ticker not in next_sells: next_sells.append(ticker)
            
            # Step E: 再平衡訊號產生 (Rebalance)
            # 每 5 天執行一次
            next_buys = []
            target_tickers = []
            
            is_rebalance_day = (rebalance_counter % 5 == 0)
            if is_rebalance_day:
                # 1. 篩選流動性
                vals = turnover_value.loc[today]
                liquid_tickers = vals[vals > v_bar_threshold].index
                
                # 2. 排名選股 (Top K)
                candidates = returns_n.loc[today, liquid_tickers].dropna()
                target_tickers = candidates.sort_values(ascending=False).head(self.top_k).index.tolist()
                
                # 3. 標記非目標持股為「待賣出」
                for ticker in holdings:
                    if ticker not in target_tickers:
                        if ticker not in next_sells:
                            holdings[ticker]['exit_reason'] = 'RebalanceOut'
                            next_sells.append(ticker)
                
                # 4. 產生買入清單 (針對 Target Tickers)
                slot_size = self.initial_capital / self.top_k
                for ticker in target_tickers:
                    if ticker not in holdings and ticker not in next_sells:
                        # 嚴格模式: 無遞補名單，僅嘗試買入 Top K
                        next_buys.append({'ticker': ticker, 'budget': slot_size})
            
            self.pending_sells = next_sells
            self.pending_buys = next_buys
            rebalance_counter += 1
            
            # 記錄當日候選股
            daily_candidates.append({'Date': today, 'Count': len(target_tickers), 'Tickers': str(target_tickers)})
            
            daily_records.append({
                'Date': today,
                'Held': len(holdings),
                'PendingSells': len(next_sells),
                'PendingBuys': len(next_buys)
            })

        # --- 4. 計算績效指標 ---
        total_days = (dates[-1] - dates[0]).days
        years = total_days / 365.25
        final_equity = equity_curve[-1]['Equity']
        cagr = (final_equity / self.initial_capital) ** (1/years) - 1 if years > 0 and final_equity > 0 else 0
        
        # Max Drawdown
        eq_series = pd.DataFrame(equity_curve).set_index('Date')['Equity']
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = dd.min()
        
        # Calmar Ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Win Rate
        wins = len([t for t in trades if t['PnL'] > 0])
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        results = {
            'CAGR': cagr,
            'MaxDD': max_dd,
            'Calmar': calmar,
            'WinRate': win_rate,
            'FinalEquity': final_equity,
            'Trades': pd.DataFrame(trades),
            'EquityCurve': pd.DataFrame(equity_curve),
            'EquityHold': pd.DataFrame(equity_hold),
            'DailyRecord': pd.DataFrame(daily_records),
            'DailyCandidate': pd.DataFrame(daily_candidates)
        }
        
        return results

    def save_results(self, results, filename='strategy_results.xlsx'):
        """儲存 Excel 結果與繪製權益曲線"""
        # 注意: 這裡只儲存單一結果至各個工作表
        with pd.ExcelWriter(filename) as writer:
            # 摘要頁
            summary = pd.DataFrame([{
                'CAGR': f"{results['CAGR']:.2%}",
                'MaxDD': f"{results['MaxDD']:.2%}",
                'Calmar': f"{results['Calmar']:.2f}",
                'WinRate': f"{results['WinRate']:.2%}",
                'FinalEquity': f"{results['FinalEquity']:,.0f}",
                'N_DAYS': self.n_days,
                'TOP_K': self.top_k,
                'V_BAR': self.v_bar,
                'TRAIL_STOP': self.trail_stop,
                'PCT_STP': self.pct_stp
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            if not results['Trades'].empty:
                results['Trades'].to_excel(writer, sheet_name='Trades', index=False)
            
            results['EquityCurve'].to_excel(writer, sheet_name='Equity_Curve', index=False)
            results['EquityHold'].to_excel(writer, sheet_name='Equity_Hold', index=False)
            results['DailyRecord'].to_excel(writer, sheet_name='Daily_Record', index=False)
            results['DailyCandidate'].to_excel(writer, sheet_name='Daily_Candidate', index=False)

        # 繪圖
        df_eq = results['EquityCurve']
        if not df_eq.empty:
            df_eq['Date'] = pd.to_datetime(df_eq['Date'])
            df_eq = df_eq.set_index('Date')
            
            plt.figure(figsize=(12, 6))
            plt.plot(df_eq['Equity'], label='Equity', color='blue')
            
            # 繪製回撤區域 (Drawdown Area)
            peak = df_eq['Equity'].cummax()
            plt.fill_between(df_eq.index, peak, df_eq['Equity'], color='red', alpha=0.1, label='Drawdown Area')
            
            plt.title('Equity Curve (權益曲線)')
            plt.legend()
            plt.grid(True)
            plt.savefig(filename.replace('.xlsx', '.png')) # Save plot with similar name
            plt.close()

class AntColonyOptimizer:
    def __init__(self, data_file):
        self.data_file = data_file
        # 參數搜索空間
        self.param_ranges = {
            'n_days': list(range(2, 5, 1)),      
            'top_k': list(range(2, 5, 1)),       
            'v_bar': list(range(5, 11, 1)),      
            'trail_stop': [i/100 for i in range(7, 12, 1)], 
            'pct_stp': [i/100 for i in range(7, 12, 1)], 
        }
        self.keys = list(self.param_ranges.keys())
        self.pheromones = {
            k: {v: 1.0 for v in vals} for k, vals in self.param_ranges.items()
        }
        self.n_ants = 200
        self.n_iterations = 3
        self.evaporation_rate = 0.2
        self.alpha = 1.0
        
        # 修改: 記錄最佳 Score
        self.best_score = -np.inf
        self.best_params = None
        self.best_result = None
        
        # 新增: 記錄前五名
        self.top_5_results = [] # 格式: (score, params, results)

    def select_param(self, key):
        """費洛蒙輪盤選擇法"""
        vals = self.param_ranges[key]
        pheros = [self.pheromones[key][v] ** self.alpha for v in vals]
        total = sum(pheros)
        probs = [p/total for p in pheros]
        return random.choices(vals, weights=probs, k=1)[0]

    def run(self):
        print(f"開始 ACO 最佳化 (目標: CAGR與Calmar並重)... 螞蟻數: {self.n_ants}, 迭代: {self.n_iterations}")
        
        # 預載入資料以提升速度
        temp_strat = Strategy(self.data_file, 10, 5, 5, 0.1, 0.1)
        shared_data = temp_strat.data
        
        for iteration in range(self.n_iterations):
            iteration_best_score = -np.inf
            iteration_best_ant = None
            
            for i in range(self.n_ants):
                params = {k: self.select_param(k) for k in self.keys}
                strat = Strategy(self.data_file, **params, data=shared_data)
                res = strat.run_backtest()
                
                # 修改: 計算綜合成績 (Score)
                cagr = res['CAGR']
                calmar = res['Calmar']
                # 權重設定: CAGR * 10 + Calmar
                # 假設 CAGR 20% (0.2) 和 Calmar 2.0 分數相當: 0.2*10 + 2.0 = 4.0
                score = (cagr * 10) + calmar 
                
                # 記錄全域最佳
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self.best_result = res
                
                # 記錄迭代最佳 (用於更新費洛蒙)
                if score > iteration_best_score:
                    iteration_best_score = score
                    iteration_best_ant = params
                
                 # 更新前五名 (避免重複參數)
                is_duplicate = False
                for _, existing_params, _ in self.top_5_results:
                    if existing_params == params:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    self.top_5_results.append((score, params, res))
                    # 排序並保留前 5
                    self.top_5_results.sort(key=lambda x: x[0], reverse=True)
                    self.top_5_results = self.top_5_results[:5]
            
            print(f"第 {iteration+1} 代最佳 Score: {iteration_best_score:.2f}, 參數: {iteration_best_ant}")
            
            # 更新費洛蒙
            for k in self.keys:
                for v in self.pheromones[k]:
                    self.pheromones[k][v] *= (1 - self.evaporation_rate)
            
            # 根據分數增加費洛蒙
            deposit = 1.0 + max(0, iteration_best_score)
            if iteration_best_ant:
                for k, v in iteration_best_ant.items():
                    self.pheromones[k][v] += deposit

        print(f"\\n最佳化完成。全域最佳 Score: {self.best_score:.2f}")
        print(f"最佳參數: {self.best_params}")
        
        print("\\n--- 前五名參數組合 ---")
        for idx, (s, p, r) in enumerate(self.top_5_results):
            print(f"Rank {idx+1}: Score={s:.2f}, CAGR={r['CAGR']:.2%}, Calmar={r['Calmar']:.2f}, Params={p}")

        return self.top_5_results

if __name__ == "__main__":
    # 自動偵測資料檔
    default_path = r'cleaned_stock_data1.xlsx'
    if os.path.exists(default_path):
        DATA_FILE = default_path
    else:
        DATA_FILE = input("請輸入 'cleaned_stock_data1.xlsx' 的完整路徑: ").strip('"').strip("'")
    print(f"使用資料檔: {DATA_FILE}")

    print("\\n請選擇模式:")
    print("1. ACO 參數最佳化 (CAGR & Calmar 並重, 輸出前五名)")
    print("2. 手動輸入參數")
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        choice = '1'
        print("Auto mode enabled. Selecting option 1.")
    else:
        try:
            choice = input("輸入選項 (1 或 2): ").strip()
        except EOFError:
            choice = "2" # Default

    if choice == '1':
        optimizer = AntColonyOptimizer(DATA_FILE)
        top_5 = optimizer.run()
        
        # 儲存前五名結果
        output_file = 'strategy_aco_top5_results.xlsx'
        
        with pd.ExcelWriter(output_file) as writer:
            # 建立 Summary Sheet
            summary_list = []
            for idx, (score, params, res) in enumerate(top_5):
                row = {
                    'Rank': idx + 1,
                    'Score': score,
                    'CAGR': res['CAGR'],
                    'MaxDD': res['MaxDD'],
                    'Calmar': res['Calmar'],
                    'WinRate': res['WinRate'],
                    'FinalEquity': res['FinalEquity'],
                    **params # 展開參數
                }
                summary_list.append(row)
                
                # 每個結果也可以存成獨立的 Sheet, 或是在這裡僅摘要
                # 為了詳細資訊，將第一名的詳細交易記錄存下來
                if idx == 0:
                    res['Trades'].to_excel(writer, sheet_name='Top1_Trades', index=False)
                    res['EquityCurve'].to_excel(writer, sheet_name='Top1_EquityCurve', index=False)
            
            summary_df = pd.DataFrame(summary_list)
            # 格式化百分比
            # (Excel 儲存時保留數值較佳，這裡不做字串格式化，讓 Excel 處理，或者用 pandas style，但這裡直接存數值)
            summary_df.to_excel(writer, sheet_name='Top5_Summary', index=False)
            
        print(f"前五名結果已儲存至 {output_file}")
        
        # 繪製第一名的圖表
        if len(top_5) > 0:
            best_res = top_5[0][2]
            df_eq = best_res['EquityCurve']
            valid_plot = False
            if not df_eq.empty:
                df_eq['Date'] = pd.to_datetime(df_eq['Date'])
                df_eq = df_eq.set_index('Date')
                
                plt.figure(figsize=(12, 6))
                plt.plot(df_eq['Equity'], label='Equity (Top 1)', color='blue')
                peak = df_eq['Equity'].cummax()
                plt.fill_between(df_eq.index, peak, df_eq['Equity'], color='red', alpha=0.1, label='Drawdown')
                plt.title(f'Equity Curve - Rank 1 (Score: {top_5[0][0]:.2f})')
                plt.legend()
                plt.grid(True)
                plt.savefig('equity_curve_top1.png')
                plt.close()
                print("第一名權益曲線已存為 equity_curve_top1.png")
        
    else:
        print("\\n--- 手動輸入參數 ---")
        try:
            # 預設值
            def_params = {'n_days': 4, 'top_k': 1, 'v_bar': 9.0, 'trail_stop': 0.1, 'pct_stp': 0.09}
            print(f"(直接按 Enter 使用這組參數: {def_params})")
            
            in_n = input(f"N_DAYS (預設 {def_params['n_days']}): ")
            n_days = int(in_n) if in_n else def_params['n_days']
            
            in_k = input(f"TOP_K (預設 {def_params['top_k']}): ")
            top_k = int(in_k) if in_k else def_params['top_k']
            
            in_v = input(f"V_BAR (預設 {def_params['v_bar']}): ")
            v_bar = float(in_v) if in_v else def_params['v_bar']
            
            in_trail = input(f"TRAIL_STOP (預設 {def_params['trail_stop']}): ")
            trail_stop = float(in_trail) if in_trail else def_params['trail_stop']
            
            in_stp = input(f"PCT_STP (預設 {def_params['pct_stp']}): ")
            pct_stp = float(in_stp) if in_stp else def_params['pct_stp']
            
            params = {
                'n_days': n_days, 'top_k': top_k, 'v_bar': v_bar,
                'trail_stop': trail_stop, 'pct_stp': pct_stp
            }
            
            print(f"\\n開始回測，參數: {params}")
            strat = Strategy(DATA_FILE, **params)
            results = strat.run_backtest()
            strat.save_results(results, filename='strategy_custom_results.xlsx')
            print("結果已儲存至 strategy_custom_results.xlsx")
            
            print(f"最終權益: {results['FinalEquity']:,.0f}")
            print(f"交易次數: {len(results['Trades'])}")
            
        except ValueError as e:
            print(f"輸入格式錯誤: {e}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import warnings
from copy import deepcopy

# Ignorwarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

class Strategy:
    def __init__(self, data_file, n_days, top_k, v_bar, trail_stop, data=None):
        """
        初始化策略
        :param data_file: 資料 Excel 路徑
        :param n_days: 過去 N 天漲幅
        :param top_k: 持有排名前 K 檔 (即 S_H)
        :param v_bar: 流動性門檻 (單位: 千萬元)
        :param trail_stop: 移動停損百分比 (例如 0.1 代表 10%)
        :param data: 預先讀取的資料 (Optional)
        """
        self.data_file = data_file
        self.n_days = int(n_days)
        self.top_k = int(top_k)
        self.v_bar = v_bar
        self.trail_stop = trail_stop
        
        # 固定參數
        self.initial_capital = 20_000_000
        self.tax_rate = 0.001
        self.slip_rate = 0.003
        
        if data is not None:
             self.data = data
        else:
             self.data = self.load_data()
        
    def load_data(self):
        """讀取並前處理資料"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"找不到檔案: {self.data_file}")
            
        xls = pd.ExcelFile(self.data_file)
        
        # 讀取 P (Adj Close) 和 Q (Volume)
        # 假設 Index 是日期
        df_p = pd.read_excel(xls, 'P', index_col=0, parse_dates=True)
        df_q = pd.read_excel(xls, 'Q', index_col=0, parse_dates=True)
        
        # 對齊資料
        common_cols = df_p.columns.intersection(df_q.columns)
        common_index = df_p.index.intersection(df_q.index)
        
        df_p = df_p.loc[common_index, common_cols]
        df_q = df_q.loc[common_index, common_cols]
        
        return {'P': df_p, 'Q': df_q}

    def run_backtest(self):
        """執行回測"""
        P = self.data['P']
        Q = self.data['Q']
        dates = P.index
        tickers = P.columns
        
        # 計算輔助數據
        # 1. 過去 N 天漲幅: (P_t / P_{t-N}) - 1
        returns_n = P.pct_change(self.n_days)
        
        # 3. 成交金額 (萬元) -> Q(張) * P(元/股) * 1000 / 10,000 = P * Q / 10
        # 需求: > V_Bar 千萬元 -> (P * Q * 1000) > V_Bar * 10,000,000
        # 即 (P * Q) > V_Bar * 10,000
        # 為了方便計算，直接算成交金額(元)
        turnover_value = P * Q * 1000
        v_bar_threshold = self.v_bar * 10_000_000

        # Calculate MA20 and MA60 for Filter 2
        ma20 = P.rolling(window=20).mean()
        ma60 = P.rolling(window=60).mean()

        # Calculate Daily Returns for Filter 1
        daily_returns = P.pct_change(1)
        
        # 回測變數
        cash = self.initial_capital
        holdings = {} # {ticker: {'shares': 0, 'entry_price': 0, 'entry_date': date, 'highest_price': 0}}
        equity_curve = []
        trades = []
        daily_records = []
        daily_candidates = []
        equity_hold = []
        
        # 每個部位的最大資金
        position_size_limit = self.initial_capital / self.top_k
        
        # 狀態
        rebalance_counter = 0
        days_since_start = 0
        
        for i in range(len(dates)):
            today = dates[i]
            if i < self.n_days:
                equity_curve.append({'Date': today, 'Equity': cash})
                equity_hold.append({'Date': today, 'Count': 0, 'Details': str({})})
                continue
                
            # 1. 計算當前權益
            current_equity = cash
            todays_prices = P.iloc[i]
            
            # 用於檢查是否上市 (價格不是 NaN)
            valid_tickers = todays_prices.dropna().index
            
            current_holdings_info = {}
            for ticker, info in list(holdings.items()):
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                     current_price = todays_prices[ticker]
                     value = info['shares'] * current_price
                     current_equity += value
                     
                     # 更新最高價 (Trailing Stop 用) - 假設以收盤價作為比較基準
                     if current_price > info['highest_price']:
                         holdings[ticker]['highest_price'] = current_price
                         
                     current_holdings_info[ticker] = info['shares']
                else:
                    # 資料缺失，可能下市，強制以最後價格平倉 (或 0)
                    # 這裡假設以最後已知價格平倉
                    pass 

            equity_curve.append({'Date': today, 'Equity': current_equity})
            equity_hold.append({'Date': today, 'Count': len(holdings), 'Details': str(current_holdings_info)})

            # 2. 檢查出場 (每日檢查) -- T日訊號，T+1執行 (這裡簡化為 T 日 Close 結算)
            
            # 初始化 pending orders (如果是第一天)
            if not hasattr(self, 'pending_sells'): self.pending_sells = []
            if not hasattr(self, 'pending_buys'): self.pending_buys = []

            # --- Step 1: Execute Pending Orders at Today's Close ---
            # 1.1 Execute Sells
            money_returned = 0
            for ticker in self.pending_sells:
                if ticker in holdings and ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    shares = holdings[ticker]['shares']
                    
                    # 考慮滑價與稅
                    # 賣出金額 = 股數 * 價格 * (1 - 滑價 - 稅)
                    revenue = shares * price * (1 - self.slip_rate - self.tax_rate)
                    money_returned += revenue
                    
                    # 紀錄 Trade
                    entry_price = holdings[ticker]['entry_price']
                    pnl = revenue - (shares * entry_price * (1 + self.slip_rate)) # 概略損益，未精確扣除買入當下成本，僅供參考
                    ret = (price * (1 - self.slip_rate - self.tax_rate)) / (entry_price * (1 + self.slip_rate)) - 1
                    
                    trades.append({
                        'Ticker': ticker,
                        'BuyDate': holdings[ticker]['entry_date'],
                        'SellDate': today,
                        'BuyPrice': entry_price,
                        'SellPrice': price,
                        'Shares': shares,
                        'PnL': pnl,
                        'Return': ret,
                        'Reason': holdings[ticker].get('exit_reason', 'Rebalance')
                    })
                    
                    del holdings[ticker]
            
            cash += money_returned
            self.pending_sells = [] # Clear

            # 1.2 Execute Buys
            # 買單： {'ticker': ticker, 'amount_budget': amount}
            # 需注意資金是否足夠 (理論上 Rebalance 會算好，但因滑價可能誤差，這裡盡量買)
            for order in self.pending_buys:
                ticker = order['ticker']
                budget = order['budget']
                
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]

                    # Filter 1: Buy Day (T+1) Price Movement Filter
                    # 若當日漲幅 > 9.5% 或 當日跌幅 < -9.5%，不做此筆交易
                    if ticker in daily_returns.columns:
                        dr = daily_returns.loc[today, ticker]
                        if not np.isnan(dr) and (dr > 0.095 or dr < -0.095):
                            # print(f"Filter 1 Triggered for {ticker} on {today}: Return {dr:.2%}")
                            continue

                    # 買入成本 = 價格 * (1 + 滑價)
                    cost_per_share = price * (1 + self.slip_rate)
                    
                    if cash >= cost_per_share * 1000: # 至少買一張
                        # 計算可買股數 (無條件捨去到張 maybe? 這裡假設可買零股或整數股，題目單位 "張" (1000)，但通常回測股數 float 較方便，這裡用 int)
                        shares_to_buy = int(budget / cost_per_share)
                        if shares_to_buy > 0:
                            actual_cost = shares_to_buy * cost_per_share
                            if cash >= actual_cost:
                                cash -= actual_cost
                                holdings[ticker] = {
                                    'shares': shares_to_buy,
                                    'entry_price': price,
                                    'entry_date': today,
                                    'highest_price': price,
                                    'exit_reason': ''
                                }
            self.pending_buys = [] # Clear

            # --- Step 2: Update Holdings High Price ---
            for ticker in holdings:
                if ticker in todays_prices and not np.isnan(todays_prices[ticker]):
                    price = todays_prices[ticker]
                    if price > holdings[ticker]['highest_price']:
                         holdings[ticker]['highest_price'] = price
            
            # --- Step 3: Generage Exit Signals (For T+1 Execution) ---
            # 每日檢查 Stops
            next_sells = []
            
            # 使用列表複製以避免迭代時修改問題，但這裡其實只是檢查
            current_held_tickers = list(holdings.keys())
            
            for ticker in current_held_tickers:
                if ticker not in todays_prices or np.isnan(todays_prices[ticker]):
                    continue
                
                price = todays_prices[ticker]
                hp = holdings[ticker]['highest_price']
                
                # Stop 1: Trail Stop
                # 下跌超過 Trail_STOP % -> imply Price < Highest * (1 - Trail_STOP)
                if price < hp * (1 - self.trail_stop):
                    holdings[ticker]['exit_reason'] = 'TrailStop'
                    if ticker not in next_sells:
                        next_sells.append(ticker)
                    continue

                # Stop 2: SMA Stop REMOVED
                        
            # --- Step 4: Generate Entry/Rebalance Signals (For T+1 Execution) ---
            # Rebalance counter
            # 第一天不 Rebalance? 通常第1天進場。
            # 假設 rebalance_counter == 0 時進場。
            # 每 5 天: 0, 5, 10
            
            next_buys = []
            
            is_rebalance_day = (rebalance_counter % 5 == 0)
            
            daily_candidates_list = []
            
            if is_rebalance_day:
                # 1. 篩選流動性
                # P * Q * 1000 > V_BAR * 10,000,000 -> Value > threshold
                # 找出今日符合流動性的股票
                # 注意: data aligned, so use todays slice
                vals = turnover_value.loc[today]
                liquid_tickers = vals[vals > v_bar_threshold].index
                
                # Filter 2: Signal Day (T) Moving Average Alignment
                # T日股價20、60日均線必須多頭排列時 (MA20 > MA60)
                ma20_today = ma20.loc[today]
                ma60_today = ma60.loc[today]
                bullish_mask = (ma20_today > ma60_today)

                # Filter liquid tickers with bullish alignment
                # Note: align indices
                valid_bullish = bullish_mask.loc[liquid_tickers]
                # valid_bullish is a boolean Series indexed by liquid_tickers
                # Filter those that are True and not NaN
                valid_bullish = valid_bullish[valid_bullish == True].index

                candidates_pool = valid_bullish

                # 2. 排名: 過去 N 天漲幅
                # returns_n.loc[today]
                # 只看 filtered tickers
                candidates = returns_n.loc[today, candidates_pool].dropna()
                
                # 取前 TOP_K
                top_candidates = candidates.sort_values(ascending=False).head(self.top_k)
                target_tickers = top_candidates.index.tolist()
                
                daily_candidates_list = target_tickers
                
                # 決定買賣
                # 策略: "每次僅持有... 前 TOP_K"
                # 意味著不在 TOP_K 的要賣掉? 
                # 0203-3.md: "每次僅持有... 再平衡: 每 5 天檢查與換股"
                # 通常意味著: 持倉中若不在 Target List，賣出。
                # 若 Target List 中沒持有的，買入 (如果有空位)。
                # 或者更嚴格: 強制換股成 Target List。
                # 考慮到 "最大持倉檔數 S_H", 且 "進場 S_B = 總資金 / S_H"。
                # 這裡假設: 強制調整組合為 Target List。
                
                # 檢查目前持倉，如果不在 target_tickers，則賣出 (除非已經因 Stop 加上去了)
                for ticker in holdings:
                    if ticker not in target_tickers:
                        if ticker not in next_sells:
                            holdings[ticker]['exit_reason'] = 'RebalanceOut'
                            next_sells.append(ticker)
                
                # 檢查 target_tickers，如果沒持有，則買入
                # 注意: 資金限制。
                # 我們預估明天賣出後會有資金。
                # 簡單模型: 
                #   預計賣出的股票 -> 釋放資金 (保守估計，忽略損益 or 假設原價? 這裡簡單假設釋放 "目前市值")
                #   現有現金
                #   計算可買清單
                
                # 估算明日可用資金 (保守)
                estimated_cash = cash
                for ticker in next_sells:
                    if ticker in holdings: 
                         # 加回目前市值 (扣一點稅費緩衝)
                         price = todays_prices[ticker] if ticker in todays_prices else 0
                         estimated_cash += holdings[ticker]['shares'] * price * (1 - self.slip_rate - self.tax_rate)
                
                # 決定買入名單
                # 依照排名優先買入
                # 每個部位預算 = initial_capital / top_k (固定金額? 題目: "進場S_B：總資金 / S_H檔")
                # 題目 "總資金 2千萬 (固定)" 可能指初始。
                # 若是指 "目前淨值 / S_H"? 
                # 通常 "Fixed Fractional" 是 Current Equity / S_H.
                # 但題目寫 "總資金 2千萬元 (固定)" 然後 "進場S_B : 總資金 / S_H"
                # 這暗示可能是 Fixed Size = 20,000,000 / S_H。不管賺賠。
                # 我們採用 Fixed Amount per slot = 20,000,000 / TOP_K.
                
                slot_size = self.initial_capital / self.top_k
                
                current_slots_used = len(holdings) - len([t for t in next_sells if t in holdings]) # 預計持有數 (排除賣出的)
                
                for ticker in target_tickers:
                    if ticker not in holdings and ticker not in next_buys: # 還沒持有 且 沒在清單
                         # 檢查是否還有空間?
                         # 嚴格來說，Rebalance 是把組合變成 Target。
                         # 如果 Target 有 K 個，我們就應該持有這 K 個。
                         # 只要資金夠。
                         
                         next_buys.append({'ticker': ticker, 'budget': slot_size})

            # 更新狀態
            self.pending_sells = next_sells
            self.pending_buys = next_buys
            
            rebalance_counter += 1
            
            # Record Candidate
            daily_candidates.append({'Date': today, 'Count': len(daily_candidates_list), 'Tickers': str(daily_candidates_list)})
            
            # Daily Record
            daily_records.append({
                'Date': today,
                'Held': len(holdings),
                'PendingSells': len(next_sells),
                'PendingBuys': len(next_buys)
            })

        # 回測結束
        total_days = (dates[-1] - dates[0]).days
        years = total_days / 365.25
        final_equity = equity_curve[-1]['Equity']
        
        cagr = (final_equity / self.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        # MaxDD
        eq_series = pd.DataFrame(equity_curve).set_index('Date')['Equity']
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = dd.min()
        
        # Calmar
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0
        
        # Win Rate
        wins = len([t for t in trades if t['PnL'] > 0])
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0
        
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
        """儲存結果"""
        with pd.ExcelWriter(filename) as writer:
            summary = pd.DataFrame([{
                'CAGR': f"{results['CAGR']:.2%}",
                'MaxDD': f"{results['MaxDD']:.2%}",
                'Calmar': f"{results['Calmar']:.2f}",
                'WinRate': f"{results['WinRate']:.2%}",
                'FinalEquity': f"{results['FinalEquity']:,.0f}",
                'N_DAYS': self.n_days,
                'TOP_K': self.top_k,
                'V_BAR': self.v_bar,
                'TRAIL_STOP': self.trail_stop
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)
            
            if not results['Trades'].empty:
                results['Trades'].to_excel(writer, sheet_name='Trades', index=False)
            
            results['EquityCurve'].to_excel(writer, sheet_name='Equity_Curve', index=False)
            results['EquityHold'].to_excel(writer, sheet_name='Equity_Hold', index=False)
            results['DailyRecord'].to_excel(writer, sheet_name='Daily_Record', index=False)
            results['DailyCandidate'].to_excel(writer, sheet_name='Daily_Candidate', index=False)

        # Plot
        df_eq = results['EquityCurve']
        if not df_eq.empty:
            df_eq['Date'] = pd.to_datetime(df_eq['Date'])
            df_eq = df_eq.set_index('Date')
            
            plt.figure(figsize=(12, 6))
            plt.plot(df_eq['Equity'], label='Equity')
            
            # Drawdown area
            peak = df_eq['Equity'].cummax()
            dd = (df_eq['Equity'] - peak) / peak
            # Fill under equity for drawdown? Usually fill dd separately or shade equity drop
            # Requirement: "含回撤陰影" usually means filling the area between high water mark and current
            plt.fill_between(df_eq.index, peak, df_eq['Equity'], color='red', alpha=0.1, label='Drawdown Area')
            
            plt.title('Equity Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig('equity_curve.png')
            plt.close()

class AntColonyOptimizer:
    def __init__(self, data_file):
        self.data_file = data_file
        # 參數範圍 (Discretized)
        self.param_ranges = {
            'n_days': list(range(3, 8, 1)),      # 5 to 60 step 5
            'top_k': list(range(2, 6, 1)),       # 3 to 10
            'v_bar': list(range(5, 20, 2)),       # 1 to 10 (千萬)
            'trail_stop': [i/100 for i in range(5, 15, 1)], # 0.05 to 0.30
        }
        
        self.keys = list(self.param_ranges.keys())
        # Pheromone matrix: key -> value_index -> pheromone_level
        self.pheromones = {
            k: {v: 1.0 for v in vals} for k, vals in self.param_ranges.items()
        }
        
        # ACO Params
        self.n_ants = 300
        self.n_iterations = 5 # 示範用，可增加
        self.evaporation_rate = 0.2
        self.alpha = 1.0 # 費洛蒙重要性
        self.best_result = None
        self.best_params = None
        # OPTIMIZATION TARGET: CALMAR RATIO
        self.best_calmar = -np.inf

    def select_param(self, key):
        """Roulette Wheel Selection based on Pheromones"""
        vals = self.param_ranges[key]
        pheros = [self.pheromones[key][v] ** self.alpha for v in vals]
        total = sum(pheros)
        probs = [p/total for p in pheros]
        return random.choices(vals, weights=probs, k=1)[0]

    def run(self):
        print(f"Starting ACO Optimization (Target: Calmar Ratio) with {self.n_ants} ants, {self.n_iterations} iterations...")
        
        # Pre-load data once to save time
        temp_strat = Strategy(self.data_file, 10, 5, 5, 0.1)
        shared_data = temp_strat.data
        
        for iteration in range(self.n_iterations):
            iteration_best_ant = None
            iteration_best_calmar = -np.inf
            
            ants_params = []
            
            # 1. Ants construct solutions
            for i in range(self.n_ants):
                params = {k: self.select_param(k) for k in self.keys}
                ants_params.append(params)
                
                # Evaluate
                strat = Strategy(self.data_file, **params, data=shared_data)
                
                res = strat.run_backtest()
                # Use Calmar for optimization
                calmar = res['Calmar']
                
                if calmar > iteration_best_calmar:
                    iteration_best_calmar = calmar
                    iteration_best_ant = params
                
                if calmar > self.best_calmar:
                    self.best_calmar = calmar
                    self.best_params = params
                    self.best_result = res
            
            print(f"Gen {iteration+1}: Best Calmar: {iteration_best_calmar:.2f}, Params: {iteration_best_ant}")
            
            # 2. Update Pheromones
            # Evaporation
            for k in self.keys:
                for v in self.pheromones[k]:
                    self.pheromones[k][v] *= (1 - self.evaporation_rate)
            
            # Deposit on best path (Iteration Best) based on Calmar
            reward = max(0, iteration_best_calmar) if iteration_best_calmar > 0 else 0 
            # Simple reward: 1 + Calmar
            deposit = 1.0 + reward
            
            if iteration_best_ant:
                for k, v in iteration_best_ant.items():
                    self.pheromones[k][v] += deposit

        print("\nOptimization Finished.")
        print(f"Best Global Calmar: {self.best_calmar:.2f}")
        print(f"Best Params: {self.best_params}")
        
        return self.best_params, self.best_result

if __name__ == "__main__":
    # Determine data file path
    default_path = r'c:\Users\user\Downloads\Python\0203-3 過去N天漲幅最高M檔\cleaned_stock_data1.xlsx'
    # Check for relative path (assuming script is in 0203-3... folder and data is in 0127-6...)
    # Using abspath to handle current working directory variations
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(current_dir, '..', '..', '0127-6  TRY1', 'cleaned_stock_data1.xlsx')
    
    if os.path.exists(default_path):
        DATA_FILE = default_path
    elif os.path.exists(relative_path):
        DATA_FILE = relative_path
    else:
        print(f"Warning: Default file not found at {default_path}")
        print(f"Warning: Relative file not found at {relative_path}")
        DATA_FILE = input("Please enter the full path to 'cleaned_stock_data1.xlsx': ").strip('"').strip("'")

    print(f"Using data file: {DATA_FILE}")

    print("\nSelect Mode:")
    print("1. ACO Optimization")
    print("2. Manual Parameter Input")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
    except EOFError:
        choice = "1" # Default to ACO if no input provided

    if choice == '2':
        # Manual Input
        print("\n--- Manual Parameter Input ---")
        try:
            n_days = int(input("Enter N_DAYS (e.g., 5): "))
            top_k = int(input("Enter TOP_K (e.g., 5): "))
            v_bar = float(input("Enter V_BAR (10 million unit, e.g., 10): "))
            trail_stop = float(input("Enter TRAIL_STOP (0.1 = 10%, e.g., 0.1): "))
            
            params = {
                'n_days': n_days,
                'top_k': top_k,
                'v_bar': v_bar,
                'trail_stop': trail_stop
            }
            
            print(f"\nRunning backtest with params: {params}")
            final_strat = Strategy(DATA_FILE, **params)
            final_results = final_strat.run_backtest()
            
            final_strat.save_results(final_results, filename='strategy_calmar_results.xlsx')
            print("Results saved to strategy_calmar_results.xlsx and equity_curve.png")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
            
    else:
        # ACO Optimization
        # 1. Run Optimization
        optimizer = AntColonyOptimizer(DATA_FILE)
        best_params, best_result = optimizer.run()
        
        # 2. Run Final Strategy with Best Params & Save
        final_strat = Strategy(DATA_FILE, **best_params)
        # final_strat.data is loaded inside
        final_results = final_strat.run_backtest() # Recalculate to be sure
        
        # Modified filename for results
        final_strat.save_results(final_results, filename='strategy_calmar_results.xlsx')
        print("Results saved to strategy_calmar_results.xlsx and equity_curve.png")

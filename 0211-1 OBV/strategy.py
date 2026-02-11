import pandas as pd
import numpy as np
import os
import time
import random
import warnings
import matplotlib.pyplot as plt

# 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 設定與常數 (Settings and Constants)
# ==========================================
class Config:
    # 檔案路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(BASE_DIR, 'cleaned_stock_data1.xlsx')
    RESULT_FILE = os.path.join(BASE_DIR, 'strategy_results.xlsx')

    # 資金管理
    INITIAL_CAPITAL = 20_000_000  # 2千萬 (固定)
    TAX_RATE = 0.001     # 交易稅 0.1%
    SLIPPAGE = 0.003     # 滑價 0.3%

    # ACO 參數
    ANT_COUNT = 50       # 螞蟻數量 (可調整)
    GENERATIONS = 10     # 世代數 (可調整)
    EVAPORATION = 0.5    # 費洛蒙揮發率
    ALPHA = 1.0          # 費洛蒙重要性因子 (控制探索與利用的平衡)

    # 策略固定參數
    RSI_PERIOD = 14      # RSI天數 (固定 14)

    # 參數範圍 (用於 ACO 探索)
    # S_H: 最大持倉檔數 (例如 2~10)
    # RE_DAYS: 再平衡天數 (例如 5~60)
    # EXIT_MA: 出場均線 (例如 5~60)
    # OBV_RANK: 每次買進時，選擇OBV前幾名的股票 (例如 1~5)
    # OBV_WINDOW: OBV增幅計算天數 (例如 2~10)
    PARAM_RANGES = {
        'S_H': list(range(2, 11, 1)),      # 2 到 10
        'RE_DAYS': list(range(5, 61, 5)),  # 5 到 60, 間隔 5
        'EXIT_MA': list(range(5, 61, 5)),  # 5 到 60, 間隔 5
        'OBV_RANK': list(range(1, 6, 1)),  # 1 到 5
        'OBV_WINDOW': list(range(2, 11, 1)) # 2 到 10
    }

# ==========================================
# 資料讀取類別 (Data Loader)
# ==========================================
class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.adj_close = None
        self.volume = None

    def load_data(self):
        print("讀取資料中... (Loading Data)")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"找不到檔案: {self.filepath}")

        xls = pd.ExcelFile(self.filepath)

        # 讀取 P (還原收盤價), Q (成交量)
        valid_sheets = [s for s in xls.sheet_names if s in ['P', 'Q']]
        if 'P' not in valid_sheets or 'Q' not in valid_sheets:
            raise ValueError("Excel 缺少 P 或 Q 工作表")

        self.adj_close = pd.read_excel(xls, 'P', index_col=0)
        self.volume = pd.read_excel(xls, 'Q', index_col=0)

        # 確保索引是 DateTime
        self.adj_close.index = pd.to_datetime(self.adj_close.index)
        self.volume.index = pd.to_datetime(self.volume.index)

        # 確保欄位一致 (取交集)
        common_cols = self.adj_close.columns.intersection(self.volume.columns)
        self.adj_close = self.adj_close[common_cols]
        self.volume = self.volume[common_cols]

        # 填補缺失值 (以防萬一，通常 cleaned data 已處理)
        self.adj_close.ffill(inplace=True)
        self.volume.fillna(0, inplace=True)

        print(f"資料讀取完成。期間: {self.adj_close.index[0].date()} 至 {self.adj_close.index[-1].date()}")
        return self.adj_close, self.volume

# ==========================================
# 策略邏輯類別 (Strategy)
# ==========================================
class Strategy:
    def __init__(self, data_close, data_volume, params):
        self.close = data_close
        self.volume = data_volume  # 單位: 張
        self.params = params

        # 參數解包
        self.S_H = params['S_H']
        self.RE_DAYS = params['RE_DAYS']
        self.EXIT_MA = params['EXIT_MA']
        self.OBV_RANK = params['OBV_RANK']
        self.OBV_WINDOW = params['OBV_WINDOW']

        # 計算指標
        self.indicators = {}
        self._calculate_indicators()

    def _calculate_rsi_wilder(self, series, period):
        delta = series.diff()

        # Wilder's Smoothing (EMA with alpha=1/period)
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # 預設 50

    def _calculate_indicators(self):
        # 1. RSI (14 days, Wilder's)
        # Apply specifically to close prices
        self.indicators['rsi'] = self.close.apply(lambda x: self._calculate_rsi_wilder(x, Config.RSI_PERIOD))

        # 2. OBV
        # OBV = Cumulative Sum of (Sign(Change) * Volume)
        # Change = Close.diff()
        # Sign: +1 if >0, -1 if <0, 0 if =0
        change = self.close.diff()
        direction = np.sign(change)
        # First element is NaN, fill with 0
        direction.iloc[0] = 0
        # Volume * direction
        # Need to align properly (pandas does this automatically by index/columns)
        obv_flow = direction * self.volume
        self.indicators['obv'] = obv_flow.cumsum()

        # 3. OBV Score (近OBV_WINDOW天增幅 * 收盤價)
        # OBV[t] - OBV[t-window]
        obv_diff = self.indicators['obv'].diff(self.OBV_WINDOW)
        # Score = Diff * Close
        self.indicators['obv_score'] = obv_diff * self.close

        # 4. EXIT_MA (出場均線)
        self.indicators['exit_ma'] = self.close.rolling(window=self.EXIT_MA).mean()

    def run_backtest(self):
        # 初始化回測變數
        capital = Config.INITIAL_CAPITAL
        # 進場金額: 總資金 / S_H (固定) -> 這裡假設每次買進都用此金額
        entry_budget = Config.INITIAL_CAPITAL / self.S_H

        positions = {} # {ticker: {'shares': int, 'cost': float, 'entry_date': date}}

        # 紀錄
        equity_curve = []
        trades = []
        daily_holdings = []
        daily_candidates = []

        dates = self.close.index
        # 從最大 days 開始，避免指標 NaN
        start_idx = max(Config.RSI_PERIOD, self.OBV_WINDOW, self.EXIT_MA, 60)

        # 為了計算 Rebalance 日期，我們可以使用相對索引
        # RE_DAYS 從回測開始算起

        for i in range(start_idx, len(dates) - 1):
            t_date = dates[i]
            next_date = dates[i+1] # T+1 日，執行交易

            # --- T日 資料與計算 ---
            current_close = self.close.loc[t_date]
            current_rsi = self.indicators['rsi'].loc[t_date]
            current_obv_score = self.indicators['obv_score'].loc[t_date]
            current_exit_ma = self.indicators['exit_ma'].loc[t_date]

            # 判斷是否為 Rebalance 日 (相對於 start_idx)
            # 題目: "每RE_DAYS日再平衡"
            is_rebalance_day = ((i - start_idx) % self.RE_DAYS == 0)

            # 候選名單 (Entry Candidates)
            entry_candidates = [] # List of tickers

            if is_rebalance_day:
                # 1. 篩選 RSI > 70
                # 2. 排序 OBV Score
                # 3. 取前 OBV_RANK 名

                # 取得 RSI > 70 的股票
                valid_candidates = current_rsi[current_rsi > 70].index.tolist()

                # 計算 Score 並排序
                scores = current_obv_score.loc[valid_candidates].dropna()
                ranked_candidates = scores.sort_values(ascending=False).index.tolist()

                # 取前 OBV_RANK 名
                entry_candidates = ranked_candidates[:self.OBV_RANK]

                daily_candidates.append({
                    'Date': t_date,
                    'Count': len(entry_candidates),
                    'Candidates': str(entry_candidates),
                    'All_Valid': len(valid_candidates)
                })
            else:
                daily_candidates.append({'Date': t_date, 'Count': 0, 'Candidates': 'No Entry Check'})

            # --- 交易決策 ---

            sell_list = [] # (ticker, reason)
            buy_list = []  # ticker

            # 1. 賣出訊號檢查 (優先權最高: 停損/訊號出場)
            # 賣出條件 (1): 跌破 EXIT_MA (T日訊號)
            # 這適用於所有持股，無論是否 Rebalance 日
            for ticker in list(positions.keys()):
                price = current_close.get(ticker)
                ma_val = current_exit_ma.get(ticker)
                if not pd.isna(price) and not pd.isna(ma_val) and price < ma_val:
                    sell_list.append((ticker, f'Exit: Price < MA{self.EXIT_MA}'))

            # 2. 進場邏輯 (僅在 Rebalance 日執行)
            # 取消 Rebalance Sell，僅執行進場買入
            if is_rebalance_day:
                current_sell_tickers = [t for t, r in sell_list]

                # 檢查 Entry Candidates: 如果未持有，且不在 sell_list -> 買進
                # 注意: 需要考慮 S_H 上限 (在執行買進時檢查)
                for ticker in entry_candidates:
                    if ticker not in positions and ticker not in current_sell_tickers:
                         buy_list.append(ticker)

            # --- 執行交易 (T+1日) ---
            next_close_prices = self.close.loc[next_date]

            # 執行賣出
            for ticker, reason in sell_list:
                if ticker in positions:
                    pos = positions[ticker]
                    exec_price = next_close_prices.get(ticker)

                    if pd.isna(exec_price) or exec_price == 0:
                        continue

                    # 滑價 (賣出價變低)
                    sell_price = exec_price * (1 - Config.SLIPPAGE)
                    # 稅
                    tax = sell_price * pos['shares'] * Config.TAX_RATE

                    revenue = sell_price * pos['shares'] - tax
                    pnl = revenue - (pos['cost'] * pos['shares'])
                    ret = (sell_price - pos['cost']) / pos['cost']

                    capital += revenue

                    trades.append({
                        'Date': next_date,
                        'Ticker': ticker,
                        'Action': 'Sell',
                        'Price': exec_price,
                        'Shares': pos['shares'],
                        'Reason': reason,
                        'PnL': pnl,
                        'Return': ret
                    })

                    del positions[ticker]

            # 執行買進
            # 注意: 需檢查資金是否足夠。以及是否已達持倉上限 S_H (雖然 Rebalance 應該控制了數量，但需防呆)
            for ticker in buy_list:
                if len(positions) >= self.S_H:
                    break # 已滿倉

                exec_price = next_close_prices.get(ticker)
                if pd.isna(exec_price) or exec_price == 0:
                    continue

                # 滑價 (買入價變高)
                buy_price = exec_price * (1 + Config.SLIPPAGE)

                # 計算股數 (無條件捨去)
                if buy_price > entry_budget:
                    continue

                shares = int(entry_budget // buy_price)
                if shares == 0:
                    continue

                cost_amt = shares * buy_price

                if capital < cost_amt:
                    continue # 現金不足

                capital -= cost_amt
                positions[ticker] = {
                    'shares': shares,
                    'cost': buy_price,
                    'entry_date': next_date
                }

                trades.append({
                    'Date': next_date,
                    'Ticker': ticker,
                    'Action': 'Buy',
                    'Price': exec_price,
                    'Shares': shares,
                    'Reason': 'Entry: Top S_H & RSI > 70',
                    'PnL': 0,
                    'Return': 0
                })

            # 結算當日權益 (T+1)
            curr_equity = capital
            holdings_info = []
            for t, pos in positions.items():
                p = next_close_prices.get(t)
                if pd.isna(p): p = pos['cost']
                mv = p * pos['shares']
                curr_equity += mv
                holdings_info.append(f"{t}({pos['shares']})")

            equity_curve.append({'Date': next_date, 'Equity': curr_equity})
            daily_holdings.append({'Date': next_date, 'Holdings_Count': len(positions), 'Details': str(holdings_info)})

        # 整理結果
        self.equity_df = pd.DataFrame(equity_curve)
        if not self.equity_df.empty:
            self.equity_df.set_index('Date', inplace=True)

        self.trades_df = pd.DataFrame(trades)
        self.daily_holdings_df = pd.DataFrame(daily_holdings)
        self.daily_candidates_df = pd.DataFrame(daily_candidates)

        return self._calculate_metrics()

    def _calculate_metrics(self):
        if self.equity_df.empty:
            return {'CAGR': -99.0, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': 0}

        initial = Config.INITIAL_CAPITAL
        final = self.equity_df['Equity'].iloc[-1]

        # Years
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        years = (end_date - start_date).days / 365.25

        if years <= 0: return {'CAGR': -0.99, 'Calmar': -99, 'MaxDD': -1.0, 'Final_Equity': final}

        # CAGR
        cagr = (final / initial) ** (1 / years) - 1

        # MaxDD
        equity = self.equity_df['Equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min() # negative value

        # Calmar
        # Avoid division by zero
        calmar = -cagr / max_dd if max_dd != 0 else 0

        return {
            'CAGR': cagr,
            'Calmar': calmar,
            'MaxDD': max_dd,
            'Final_Equity': final
        }

# ==========================================
# 螞蟻演算法 (ACO) 類別
# ==========================================
class AntColonyOptimizer:
    def __init__(self, data_close, data_volume):
        self.close = data_close
        self.volume = data_volume
        self.best_solution = None
        self.best_fitness = -np.inf

        # 初始化費洛蒙
        self.pheromones = {}
        for param, values in Config.PARAM_RANGES.items():
            self.pheromones[param] = {v: 1.0 for v in values}

    def _select_value(self, param):
        values = Config.PARAM_RANGES[param]
        # 使用 Config.ALPHA 作為費洛蒙重要性因子
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in values]
        total = sum(probs)
        if total == 0:
            probs = [1.0 / len(values)] * len(values)
        else:
            probs = [p / total for p in probs]
        return np.random.choice(values, p=probs)

    def run(self):
        print("\n開始 ACO 最佳化... (Starting ACO)")

        for gen in range(Config.GENERATIONS):
            gen_best_fitness = -np.inf
            gen_best_ant = None
            ants_results = []

            print(f"--- 世代 {gen + 1} / {Config.GENERATIONS} ---")

            for ant in range(Config.ANT_COUNT):
                # 建構解
                params = {}
                for k in Config.PARAM_RANGES.keys():
                    params[k] = self._select_value(k)

                # 執行策略
                strat = Strategy(self.close, self.volume, params)
                metrics = strat.run_backtest()
                fitness = metrics['CAGR'] # 目標: 最大化 CAGR

                ants_results.append((params, fitness, metrics))

                # 更新世代最佳
                if fitness > gen_best_fitness:
                    gen_best_fitness = fitness
                    gen_best_ant = params

                # 更新全域最佳
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = params
                    print(f"  [新紀錄] Ant {ant}: CAGR={fitness:.2%}, Calmar={metrics['Calmar']:.2f}")
                    print(f"  Params: {params}")

            # 費洛蒙更新
            # 蒸發
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - Config.EVAPORATION)

            # 堆積 (菁英策略: 強化前 50%)
            sorted_ants = sorted(ants_results, key=lambda x: x[1], reverse=True)
            top_ants = sorted_ants[:max(1, len(sorted_ants)//2)]

            for params, fitness, _ in top_ants:
                # 獎勵機制: 正報酬給予獎勵，負報酬給予微小獎勵(鼓勵探索但偏好正向)
                deposit = max(0.01, fitness)
                for k, v in params.items():
                    self.pheromones[k][v] += deposit

            print(f"  世代最佳: CAGR={gen_best_fitness:.2%}")

        print("\nACO 完成。")
        print(f"最佳參數: {self.best_solution}")
        print(f"最佳 CAGR: {self.best_fitness:.2%}")
        return self.best_solution

# ==========================================
# 主程式 (Main)
# ==========================================
def main():
    # 執行模式選擇
    # 可以透過簡單的變數切換
    # RUN_MODE = 'ACO' # 執行最佳化
    RUN_MODE = 'MANUAL' # 手動輸入參數
    # 如果要改為預設 ACO，請修改此處

    # 由於題目要求 "內有參數輸入旗標：手動輸入參數，或者ACO回測"
    # 這裡預設為 ACO 以符合 "跑出最佳化參數後" 的流程，或者設為 Manual 方便測試
    # 我將預設為 ACO，但如果想測試特定參數可改 Manual
    RUN_MODE = 'ACO'

    # 手動參數設定 (若 RUN_MODE = 'MANUAL')
    MANUAL_PARAMS = {
        'S_H': 5,
        'RE_DAYS': 20,
        'EXIT_MA': 20,
        'OBV_RANK': 3,
        'OBV_WINDOW': 5
    }

    # 1. 讀取資料
    try:
        loader = DataLoader(Config.DATA_FILE)
        close_data, volume_data = loader.load_data()
    except Exception as e:
        print(f"錯誤: 無法讀取資料 - {e}")
        return

    best_params = {}

    if RUN_MODE == 'ACO':
        # 2. 執行 ACO 找最佳參數
        optimizer = AntColonyOptimizer(close_data, volume_data)
        best_params = optimizer.run()
    else:
        print(f"使用手動參數: {MANUAL_PARAMS}")
        best_params = MANUAL_PARAMS

    # 3. 使用最佳參數跑最後一次策略並輸出詳細報告
    print("\n使用參數產生最終報告...")
    final_strat = Strategy(close_data, volume_data, best_params)
    metrics = final_strat.run_backtest()

    # 4. 輸出 Excel
    print(f"儲存結果至 {Config.RESULT_FILE}...")
    try:
        with pd.ExcelWriter(Config.RESULT_FILE, engine='openpyxl') as writer:
            final_strat.trades_df.to_excel(writer, sheet_name='Trades')
            final_strat.equity_df.to_excel(writer, sheet_name='Equity_Curve')
            final_strat.daily_holdings_df.to_excel(writer, sheet_name='Equity_Hold')
            final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Candidate')
            final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Record') # 重複使用 Candidate sheet 作為 Record，或需另外生成

            # Summary
            summary_data = {
                'Metric': ['CAGR', 'MaxDD', 'Calmar', 'Final Equity', 'Best Params'],
                'Value': [
                    metrics['CAGR'],
                    metrics['MaxDD'],
                    metrics['Calmar'],
                    metrics['Final_Equity'],
                    str(best_params)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        print("Excel 儲存完成。")

        # 畫權益曲線圖
        if not final_strat.equity_df.empty:
            plt.figure(figsize=(10, 6))
            equity_series = final_strat.equity_df['Equity']
            plt.plot(equity_series.index, equity_series, label='Equity')

            # 回撤陰影 (Drawdown)
            cummax = equity_series.cummax()
            dd = (equity_series - cummax) / cummax
            # 畫在副軸? 或直接畫 Equity
            # 題目: "含回撤陰影"。通常指 High-Water Mark 填充
            plt.fill_between(equity_series.index, equity_series, cummax, color='red', alpha=0.1, label='Drawdown Area')

            plt.title(f"Equity Curve (CAGR: {metrics['CAGR']:.2%})")
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            output_png = os.path.join(Config.BASE_DIR, 'equity_curve.png')
            plt.savefig(output_png)
            print(f"權益曲線圖已儲存為 {output_png}")
        else:
            print("無交易數據，無法繪圖。")

    except Exception as e:
        print(f"儲存結果失敗: {e}")

if __name__ == '__main__':
    main()

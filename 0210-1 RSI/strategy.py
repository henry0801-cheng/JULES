import pandas as pd
import numpy as np
import os
import time
import random
import calendar
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 忽略 FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================================
# 設定與常數 (Settings and Constants)
# ==========================================
class Config:
    # 執行模式: 'ACO' (最佳化) 或 'MANUAL' (手動參數)
    RUN_MODE = 'ACO'
    # RUN_MODE = 'MANUAL'

    # 檔案路徑
    DATA_FILE = 'cleaned_stock_data2.xlsx'
    RESULT_FILE = 'strategy_results.xlsx'

    # 資金管理
    INITIAL_CAPITAL = 20_000_000  # 2千萬
    TAX_RATE = 0.001     # 交易稅 0.1%
    SLIPPAGE = 0.003     # 滑價 0.3%

    # ACO 參數 (僅在 RUN_MODE='ACO' 時使用)
    ANT_COUNT = 50       # 螞蟻數量
    GENERATIONS = 10     # 世代數
    EVAPORATION = 0.5    # 費洛蒙揮發率
    ALPHA = 1.0          # 費洛蒙重要性因子

    # 參數範圍 (用於 ACO 探索)
    # S_H: 最大持倉檔數
    # V_BAR: 20日均量是60日均量的倍數門檻
    # exit_ma_period: 出場均線週期 (例如 5, 10, 20, 60)
    PARAM_RANGES = {
        'S_H': list(range(2, 6, 1)),             # 2, 3, 4, 5
        'V_BAR': [round(x, 1) for x in np.arange(1.0, 5.0, 0.2)], # 1.0 ~ 4.8
        'exit_ma_period': list(range(5, 65, 5))  # 5, 10, ..., 60
    }

    # 手動參數 (僅在 RUN_MODE='MANUAL' 時使用)
    MANUAL_PARAMS = {
        'S_H': 3,
        'V_BAR': 2.0,
        'exit_ma_period': 20
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
            # 嘗試在當前目錄尋找
            if os.path.exists(os.path.join(os.getcwd(), self.filepath)):
                 self.filepath = os.path.join(os.getcwd(), self.filepath)
            else:
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
        self.V_BAR = params['V_BAR']
        self.exit_ma_period = params['exit_ma_period']

        # 計算指標
        self.indicators = {}
        self._calculate_indicators()

    def _calculate_indicators(self):
        # 1. 價格均線 (Price MA)
        self.indicators['ma20'] = self.close.rolling(window=20).mean()
        self.indicators['ma60'] = self.close.rolling(window=60).mean()
        self.indicators['exit_ma'] = self.close.rolling(window=self.exit_ma_period).mean()

        # 2. 成交量均線 (Volume MA)
        self.indicators['ma20_vol'] = self.volume.rolling(window=20).mean()
        self.indicators['ma60_vol'] = self.volume.rolling(window=60).mean()

    def run_backtest(self):
        # 初始化回測變數
        capital = Config.INITIAL_CAPITAL
        # 固定每檔分配金額 (總資金 / S_H)
        entry_budget = Config.INITIAL_CAPITAL / self.S_H

        positions = {} # {ticker: {'shares': int, 'cost': float, 'entry_date': date}}

        # 紀錄
        equity_curve = [] # [{'Date': date, 'Equity': float}]
        trades = [] # List of trade records
        daily_holdings = [] # List of daily holdings
        daily_candidates = [] # 符合條件股票數

        dates = self.close.index
        # 從最大 days 開始，避免指標 NaN
        start_idx = max(60, self.exit_ma_period)

        for i in range(start_idx, len(dates) - 1):
            t_date = dates[i]
            next_date = dates[i+1] # T+1 日，執行交易

            # --- T日 資料與訊號計算 ---
            current_close = self.close.loc[t_date]
            current_ma20 = self.indicators['ma20'].loc[t_date]
            current_ma60 = self.indicators['ma60'].loc[t_date]
            current_ma20_vol = self.indicators['ma20_vol'].loc[t_date]
            current_ma60_vol = self.indicators['ma60_vol'].loc[t_date]
            current_exit_ma = self.indicators['exit_ma'].loc[t_date]

            # 1. 找出潛在進場標的 (Candidates)
            # 條件:
            # (1) 股價 20MA > 60MA (多頭排列)
            # (2) 20MA量 > V_BAR * 60MA量

            # 先篩選符合條件的股票
            # 向量化運算加速
            cond1 = (current_ma20 > current_ma60)
            cond2 = (current_ma20_vol > current_ma60_vol * self.V_BAR)
            # 排除無效值
            valid_mask = current_close.notna() & current_ma20.notna() & current_ma60.notna() & \
                         current_ma20_vol.notna() & current_ma60_vol.notna()

            candidates_mask = cond1 & cond2 & valid_mask
            candidates_list = candidates_mask[candidates_mask].index.tolist()

            # 排序: 為了決定當多個標的同時符合時優先選誰。
            # 這裡使用 "成交量爆發程度" (MA20_Vol / MA60_Vol) 作為強度指標。
            # 強度越高越優先。
            candidates_scores = []
            for ticker in candidates_list:
                vol20 = current_ma20_vol.get(ticker)
                vol60 = current_ma60_vol.get(ticker)
                if vol60 > 0:
                    score = vol20 / vol60
                else:
                    score = 0
                candidates_scores.append((ticker, score))

            # 由高到低排序
            candidates_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_candidates = [x[0] for x in candidates_scores]

            daily_candidates.append({'Date': t_date, 'Count': len(sorted_candidates), 'Candidates': str(sorted_candidates)})

            # --- 交易執行 (T+1日) ---
            next_close = self.close.loc[next_date]

            # 2. 賣出檢查 (Exit Conditions)
            # 出場條件: 股價跌破 Exit_MA
            # 訊號產生於 T 日 (current_close < current_exit_ma) -> T+1 賣出

            stocks_to_sell = []

            for ticker in list(positions.keys()):
                pos = positions[ticker]
                price_t = current_close.get(ticker)
                exit_ma_t = current_exit_ma.get(ticker)

                # 檢查出場條件
                should_exit = False
                reason = ""

                if pd.isna(price_t) or pd.isna(exit_ma_t):
                    # 資料不足無法判斷，暫不動作，或者如果完全沒報價了(下市?)
                    # 這裡假設若無報價則不出場 (或等到有報價)
                    pass
                elif price_t < exit_ma_t:
                    should_exit = True
                    reason = f"Exit: Price({price_t:.2f}) < MA{self.exit_ma_period}({exit_ma_t:.2f})"

                if should_exit:
                    stocks_to_sell.append((ticker, reason))

            # 執行賣出
            for ticker, reason in stocks_to_sell:
                if ticker in positions:
                    pos = positions[ticker]
                    exec_price = next_close.get(ticker)

                    if pd.isna(exec_price) or exec_price == 0:
                        continue # 無法交易

                    # 賣出計算: 價格 * (1 - 滑價) - 稅
                    sell_price = exec_price * (1 - Config.SLIPPAGE)
                    tax = sell_price * pos['shares'] * Config.TAX_RATE
                    revenue = sell_price * pos['shares'] - tax

                    cost_basis = pos['cost'] * pos['shares']
                    pnl = revenue - cost_basis
                    ret = (revenue - cost_basis) / cost_basis

                    capital += revenue

                    trades.append({
                        'Date': next_date,
                        'Ticker': ticker,
                        'Action': 'Sell',
                        'Price': exec_price,
                        'Shares': pos['shares'],
                        'Reason': reason,
                        'PnL': int(pnl),
                        'Return': ret
                    })

                    del positions[ticker]

            # 3. 買進檢查 (Entry Conditions)
            # 依序檢查候選名單，若有空位則買進

            for ticker in sorted_candidates:
                if ticker in positions:
                    continue # 已持有

                if len(positions) >= self.S_H:
                    break # 已滿倉

                # 執行買進
                exec_price = next_close.get(ticker)
                if pd.isna(exec_price) or exec_price == 0:
                    continue

                # 買進計算: 價格 * (1 + 滑價)
                buy_price = exec_price * (1 + Config.SLIPPAGE)

                if buy_price > entry_budget:
                    continue # 買不起一張

                shares = int(entry_budget // buy_price)
                if shares == 0:
                    continue

                cost_amt = shares * buy_price

                # 檢查現金 (理論上 entry_budget 是基於初始資金分配，但實務上需有足夠現金)
                # 這裡使用 capital (現金池) 檢查
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
                    'Reason': f"Entry: MA20>MA60, Vol>{self.V_BAR}x",
                    'PnL': 0,
                    'Return': 0
                })

            # 4. 結算當日權益 (T+1)
            curr_equity = capital
            holdings_info = []
            for t, pos in positions.items():
                p = next_close.get(t)
                if pd.isna(p): p = pos['cost'] # 若無價格，用成本估算
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
            return {'CAGR': -0.99, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': 0}

        initial = Config.INITIAL_CAPITAL
        final = self.equity_df['Equity'].iloc[-1]

        # Years
        start_date = self.equity_df.index[0]
        end_date = self.equity_df.index[-1]
        years = (end_date - start_date).days / 365.25

        if years <= 0: return {'CAGR': -0.99, 'Calmar': -99.0, 'MaxDD': -1.0, 'Final_Equity': final}

        # CAGR
        cagr = (final / initial) ** (1 / years) - 1

        # MaxDD
        equity = self.equity_df['Equity']
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = drawdown.min() # negative value

        # Calmar
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
        probs = [self.pheromones[param][v] ** Config.ALPHA for v in values]
        total = sum(probs)
        if total == 0:
            probs = [1.0/len(values)] * len(values)
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

            # 費洛蒙更新 (蒸發 + 堆積)
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - Config.EVAPORATION)

            # 菁英強化 (Top 50%)
            sorted_ants = sorted(ants_results, key=lambda x: x[1], reverse=True)
            top_ants = sorted_ants[:max(1, len(sorted_ants)//2)]

            for params, fitness, _ in top_ants:
                deposit = max(0, fitness) if fitness > 0 else 0.01
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
    # 1. 讀取資料
    # 注意: 根據指示，使用 cleaned_stock_data2.xlsx
    # 假設檔案在當前目錄或相對路徑
    loader = DataLoader(Config.DATA_FILE)
    try:
        close_data, volume_data = loader.load_data()
    except Exception as e:
        print(f"資料讀取失敗: {e}")
        return

    best_params = {}

    # 2. 根據模式執行
    if Config.RUN_MODE == 'ACO':
        print(f"目前模式: ACO (自動最佳化)")
        optimizer = AntColonyOptimizer(close_data, volume_data)
        best_params = optimizer.run()
    elif Config.RUN_MODE == 'MANUAL':
        print(f"目前模式: MANUAL (手動參數)")
        best_params = Config.MANUAL_PARAMS
        print(f"使用參數: {best_params}")
    else:
        print(f"未知的模式: {Config.RUN_MODE}")
        return

    # 3. 使用最佳參數跑最後一次策略並輸出詳細報告
    print("\n使用最終參數產生詳細報告...")
    final_strat = Strategy(close_data, volume_data, best_params)
    metrics = final_strat.run_backtest()

    # 4. 輸出 Excel
    try:
        with pd.ExcelWriter(Config.RESULT_FILE, engine='openpyxl') as writer:
            final_strat.trades_df.to_excel(writer, sheet_name='Trades')
            final_strat.equity_df.to_excel(writer, sheet_name='Equity_Curve')
            final_strat.daily_holdings_df.to_excel(writer, sheet_name='Equity_Hold')
            final_strat.daily_candidates_df.to_excel(writer, sheet_name='Daily_Candidate')

            # Summary
            summary_data = {
                'Metric': ['CAGR', 'MaxDD', 'Calmar', 'Final Equity', 'Best Params', 'Run Mode'],
                'Value': [
                    metrics['CAGR'],
                    metrics['MaxDD'],
                    metrics['Calmar'],
                    metrics['Final_Equity'],
                    str(best_params),
                    Config.RUN_MODE
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        print(f"結果已儲存至 {Config.RESULT_FILE}")

        # 畫權益曲線圖
        if not final_strat.equity_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(final_strat.equity_df.index, final_strat.equity_df['Equity'], label='Equity')
            plt.title(f"Equity Curve (CAGR: {metrics['CAGR']:.2%})")
            plt.xlabel('Date')
            plt.ylabel('Equity')
            plt.legend()
            plt.grid(True)
            plt.savefig('equity_curve.png')
            print("權益曲線圖已儲存為 equity_curve.png")
        else:
            print("無交易紀錄，無法繪製權益曲線圖。")

    except Exception as e:
        print(f"結果輸出失敗: {e}")

if __name__ == '__main__':
    main()

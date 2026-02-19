import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import os
import random

# ==========================================
# Configuration and Constants
# ==========================================
INITIAL_CAPITAL = 20_000_000
TAX_RATE = 0.001
SLIPPAGE = 0.003
ACO_GENERATIONS = 5  # Reduced for reasonable runtime in this environment, can be increased
ACO_ANTS = 10        # Number of ants per generation
ACO_ALPHA = 1.0      # Pheromone importance
ACO_EVAPORATION = 0.1
ACO_Q = 1.0

# Parameter Ranges (Discrete)
PARAM_RANGES = {
    'S_H': [3, 4, 5, 6, 7, 8, 9, 10],
    'RSI_DAY': [5, 10, 14, 20, 25, 30, 40, 50, 60],
    'V_BAR': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

# ==========================================
# Data Loading
# ==========================================
def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        xls = pd.ExcelFile(filepath)
        df_P = pd.read_excel(xls, 'P', index_col=0, parse_dates=True)
        df_Q = pd.read_excel(xls, 'Q', index_col=0, parse_dates=True)
        # df_F = pd.read_excel(xls, 'F', index_col=0, parse_dates=True) # Not used in strategy logic
        print("Data loaded successfully.")
        return df_P, df_Q
    except FileNotFoundError:
        # Fallback to alternative filename if specific one is missing
        alt_path = filepath.replace('cleaned_stock_data2.xlsx', 'cleaned_stock_data1.xlsx')
        if os.path.exists(alt_path):
             print(f"File not found. Trying {alt_path}...")
             return load_data(alt_path)
        else:
            raise

# ==========================================
# Indicators
# ==========================================
def calculate_rsi_ema(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    # Use EWM with span=period which corresponds to EMA
    # Note: Standard RSI often uses com=period-1 (Wilder's).
    # "EMA算法" implies standard EMA (span=period).
    ma_up = up.ewm(span=period, adjust=False).mean()
    ma_down = down.ewm(span=period, adjust=False).mean()

    rsi = 100 - (100 / (1 + ma_up / ma_down))
    return rsi

def calculate_indicators(df_P, df_Q, rsi_day):
    # RSI
    print(f"Calculating RSI({rsi_day})...")
    rsi_df = df_P.apply(lambda x: calculate_rsi_ema(x, rsi_day))

    # MA20 Volume
    print("Calculating MA20 Volume...")
    vol_ma20 = df_Q.rolling(window=20).mean()

    # Transaction Amount (MA20_Vol * 1000 * Price)
    # Volume is in "張" (1000 shares), so multiply by 1000.
    # We use Close price for calculation of T-day signal
    trans_amt = vol_ma20 * 1000 * df_P

    return rsi_df, trans_amt

# ==========================================
# Strategy Logic
# ==========================================
class Strategy:
    def __init__(self, df_P, rsi_df, trans_amt, params):
        self.df_P = df_P
        self.rsi_df = rsi_df
        self.trans_amt = trans_amt
        self.S_H = params['S_H']
        self.V_BAR = params['V_BAR']
        self.capital_per_stock = INITIAL_CAPITAL / self.S_H

    def run_backtest(self):
        dates = self.df_P.index
        equity_curve = []
        trades = []
        daily_record = []
        daily_candidate = []
        equity_hold = []

        cash = INITIAL_CAPITAL
        holdings = {} # {ticker: quantity}

        # Portfolio value history
        portfolio_value_series = pd.Series(index=dates, dtype=float)

        # Determine Rebalance Days
        # "每5天再平衡一次".
        # We can iterate and check index % 5 == 0.

        next_holdings = {}
        rebalance_signal = False

        # We start from day 20+ to allow indicators to warm up, or just handle NaNs
        start_idx = 20

        for i in range(start_idx, len(dates)):
            date = dates[i]
            prev_date = dates[i-1] if i > 0 else date

            # Current Prices (Open for execution, Close for valuation/signal)
            # Signal is calculated on Day T (using Close T), Executed on T+1 (using Open T+1)
            # So on Day T (loop i), we calculate target for T+1.
            # Orders from T-1 calculation are executed today (Day T) at Open.

            # --- Execution Phase (at Open of Day T) ---
            # We assume orders were generated at T-1 and stored in `next_holdings`
            # But wait, logic says "T日計算訊號，T+1日執行".
            # So at index i (Day T), we execute orders determined at i-1 (Day T-1).

            current_prices_open = self.df_P.iloc[i] # Use Close as proxy if Open not avail, but we should use Open?
            # The loaded data only has 'P' (Adjusted Close). It doesn't seem to have Open.
            # "P 工作表：還原收盤價".
            # If we don't have Open, we must use Close of T (slippage accounts for it somewhat) or Close of T-1?
            # Standard backtest with only Close: Execute at Close of T (Signal T) or Open of T+1 (approx by Close T or Close T+1).
            # "T+1日執行". Without Open data, I will use Close of T+1 as the execution price (Next Close).
            # Or assume P is Close and we trade at P[i] which is T.
            # If Signal is T, Execute T+1. Then execution price is P[i] (Close of T+1) or we need Open.
            # Given only P (Close) is available, I will use P[i] (Close of Day T) as execution price for signals from T-1?
            # No, if Signal T -> Execute T+1.
            # If I am at loop i (Day T), I can execute orders from T-1 using P[i] (Close T).
            # This means we trade at Close of T+1.

            # Actually, let's assume P is the price we can trade at.
            # If we signal at T (using P[T]), and execute T+1 (at P[T+1]), we use P[i] for execution of orders from i-1.

            current_price = self.df_P.iloc[i]

            # Execute pending portfolio changes
            if i > start_idx and rebalance_signal:
                # Sell everything not in next_holdings or reduce quantity
                # Buy new holdings

                # Sell Logic
                stocks_to_sell = [s for s in holdings if s not in next_holdings]
                for stock in stocks_to_sell:
                    price = current_price[stock]
                    if pd.isna(price): continue
                    qty = holdings[stock]
                    revenue = qty * price * (1 - SLIPPAGE - TAX_RATE)
                    cash += revenue
                    del holdings[stock]
                    trades.append({
                        'Date': date, 'Ticker': stock, 'Action': 'Sell',
                        'Price': price, 'Qty': qty, 'Revenue': revenue, 'Reason': 'Rebalance Out'
                    })

                # Buy/Rebalance Logic
                # "進場S_B：總資金 / S_H檔" -> Fixed amount per slot
                target_value_per_slot = INITIAL_CAPITAL / self.S_H

                for stock in next_holdings:
                    if stock not in holdings:
                        price = current_price[stock]
                        if pd.isna(price) or price == 0: continue

                        cost = price * (1 + SLIPPAGE)
                        qty = int(target_value_per_slot / cost)

                        if cash >= qty * cost and qty > 0:
                            cash -= qty * cost
                            holdings[stock] = qty
                            trades.append({
                                'Date': date, 'Ticker': stock, 'Action': 'Buy',
                                'Price': price, 'Qty': qty, 'Cost': qty * cost, 'Reason': 'Rebalance In'
                            })

                rebalance_signal = False # Reset

            # --- Valuation Phase (Close of Day T) ---
            port_value = cash
            equity_hold_record = []
            for stock, qty in holdings.items():
                price = current_price[stock]
                if not pd.isna(price):
                    port_value += qty * price
                equity_hold_record.append(f"{stock}:{qty}")

            portfolio_value_series[date] = port_value
            equity_hold.append({'Date': date, 'Holdings': str(equity_hold_record), 'Count': len(holdings)})

            # --- Signal Generation Phase (End of Day T) ---
            # Check if tomorrow (i+1) should be an execution day.
            # We rebalance every 5 days.
            # Let's say we check condition at i. If condition met, set rebalance_signal for i+1.

            if i % 5 == 0:
                # Rebalance Calculation
                # 1. Filter
                # "20日均量換算成交金額 > V_BAR千萬元"
                # Signal calculated on T (today i)
                candidates = self.trans_amt.iloc[i] > (self.V_BAR * 10_000_000)
                candidate_stocks = candidates[candidates].index.tolist()

                # 2. Rank by RSI
                # "選擇RSI_DAY日RSI最高S_H檔"
                current_rsi = self.rsi_df.iloc[i][candidate_stocks]
                # Filter out NaNs in RSI
                current_rsi = current_rsi.dropna()

                top_stocks = current_rsi.sort_values(ascending=False).head(self.S_H).index.tolist()

                next_holdings = {s: 1 for s in top_stocks} # Value doesn't matter, just set
                rebalance_signal = True

                daily_candidate.append({'Date': date, 'Count': len(candidate_stocks), 'Stocks': str(top_stocks)})
                daily_record.append({'Date': date, 'Action': 'Signal Gen', 'Target': str(top_stocks)})
            else:
                daily_record.append({'Date': date, 'Action': 'Hold', 'Target': 'Same'})

        # End of loop
        self.results = {
            'equity_curve': portfolio_value_series,
            'trades': pd.DataFrame(trades),
            'equity_hold': pd.DataFrame(equity_hold),
            'daily_record': pd.DataFrame(daily_record),
            'daily_candidate': pd.DataFrame(daily_candidate)
        }
        return self.results

    def evaluate(self):
        # Calculate CAGR and Calmar
        equity = self.results['equity_curve'].dropna()
        if equity.empty: return -999, -999

        days = (equity.index[-1] - equity.index[0]).days
        years = days / 365.25
        if years == 0: return -999, -999

        final_return = equity.iloc[-1] / INITIAL_CAPITAL
        cagr = (final_return ** (1/years)) - 1

        # Max DD
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        max_dd = dd.min() # negative value

        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        return cagr, calmar

# ==========================================
# Ant Colony Optimization
# ==========================================
class AntColonyOptimizer:
    def __init__(self, df_P, df_Q):
        self.df_P = df_P
        self.df_Q = df_Q
        # Pheromones: Dict of Dict. {Param: {Value: PheromoneLevel}}
        self.pheromones = {}
        for param, values in PARAM_RANGES.items():
            self.pheromones[param] = {v: 1.0 for v in values}

        self.best_score = -float('inf')
        self.best_params = None
        self.best_metrics = None

    def select_value(self, param):
        values = PARAM_RANGES[param]
        phero_levels = np.array([self.pheromones[param][v] for v in values])
        # Probability = Pheromone^Alpha / Sum(...)
        probs = phero_levels ** ACO_ALPHA
        probs = probs / probs.sum()
        return np.random.choice(values, p=probs)

    def optimize(self):
        print("Starting ACO Optimization...")

        # Pre-calculate RSI for all possible RSI_DAYs to save time?
        # No, that might consume too much memory or time. Calculate on demand or cache.
        # Given small range of RSI_DAY, we can pre-calculate.
        rsi_cache = {}
        print("Pre-calculating RSI cache...")
        for rsi_day in PARAM_RANGES['RSI_DAY']:
            rsi_df, _ = calculate_indicators(self.df_P, self.df_Q, rsi_day)
            rsi_cache[rsi_day] = rsi_df

        # TransAmt is same for all, depend on V_BAR? No, TransAmt is calc once, Filter depends on V_BAR.
        # TransAmt only depends on MA20 Vol and Price. Independent of params.
        _, trans_amt_base = calculate_indicators(self.df_P, self.df_Q, 20) # Dummy rsi

        for gen in range(ACO_GENERATIONS):
            print(f"\n--- Generation {gen+1}/{ACO_GENERATIONS} ---")
            gen_solutions = []

            for ant in range(ACO_ANTS):
                # Construct Solution
                params = {
                    'S_H': self.select_value('S_H'),
                    'RSI_DAY': self.select_value('RSI_DAY'),
                    'V_BAR': self.select_value('V_BAR')
                }

                # Evaluate
                rsi_df = rsi_cache[params['RSI_DAY']]
                strat = Strategy(self.df_P, rsi_df, trans_amt_base, params)
                strat.run_backtest()
                cagr, calmar = strat.evaluate()

                score = cagr + calmar # Objective Function

                gen_solutions.append({
                    'params': params,
                    'score': score,
                    'cagr': cagr,
                    'calmar': calmar
                })

            # Update Pheromones
            # Evaporation
            for param in self.pheromones:
                for v in self.pheromones[param]:
                    self.pheromones[param][v] *= (1 - ACO_EVAPORATION)

            # Deposit (Best ant of gen, or all weighted?)
            # Usually Global Best or Iteration Best. Let's use Iteration Best + Global Best.
            gen_solutions.sort(key=lambda x: x['score'], reverse=True)
            best_gen = gen_solutions[0]

            print(f"Gen {gen+1} Best: {best_gen['params']}, CAGR: {best_gen['cagr']:.2%}, Calmar: {best_gen['calmar']:.2f}")

            if best_gen['score'] > self.best_score:
                self.best_score = best_gen['score']
                self.best_params = best_gen['params']
                self.best_metrics = (best_gen['cagr'], best_gen['calmar'])

            # Deposit Pheromone on best gen path
            deposit = ACO_Q * max(0, best_gen['score']) # Ensure positive
            for p_name, p_val in best_gen['params'].items():
                self.pheromones[p_name][p_val] += deposit

        print(f"\nOptimization Complete. Best Params: {self.best_params}")
        print(f"Best CAGR: {self.best_metrics[0]:.2%}, Calmar: {self.best_metrics[1]:.2f}")
        return self.best_params

# ==========================================
# Main
# ==========================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'cleaned_stock_data2.xlsx')

    # Load Data
    df_P, df_Q = load_data(data_path)

    # Run ACO
    optimizer = AntColonyOptimizer(df_P, df_Q)
    best_params = optimizer.optimize()

    # Run Final Strategy with Best Params
    print("\nRunning Final Strategy with Best Parameters...")
    rsi_day = best_params['RSI_DAY']
    rsi_df, trans_amt = calculate_indicators(df_P, df_Q, rsi_day)

    strat = Strategy(df_P, rsi_df, trans_amt, best_params)
    results = strat.run_backtest()
    cagr, calmar = strat.evaluate()

    # Export Results
    print("Exporting Results...")
    output_path = os.path.join(base_dir, 'strategy_results.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        results['trades'].to_excel(writer, sheet_name='Trades')
        results['equity_curve'].to_excel(writer, sheet_name='Equity_Curve')
        results['equity_hold'].to_excel(writer, sheet_name='Equity_Hold')
        results['daily_record'].to_excel(writer, sheet_name='Daily_Record')
        results['daily_candidate'].to_excel(writer, sheet_name='Daily_Candidate')

        summary = pd.DataFrame([{
            'CAGR': cagr,
            'Calmar': calmar,
            'Best Params': str(best_params)
        }])
        summary.to_excel(writer, sheet_name='Summary')

    print(f"Results saved to {output_path}")

    # Plot Equity Curve
    plt.figure(figsize=(10, 6))
    equity = results['equity_curve']
    equity.plot(label='Equity')

    # Draw Drawdown
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    plt.fill_between(equity.index, equity, cummax, color='red', alpha=0.1, label='Drawdown')

    plt.title(f"Equity Curve (CAGR: {cagr:.2%}, Calmar: {calmar:.2f})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(base_dir, 'equity_curve.png'))
    print("Equity curve saved.")

if __name__ == "__main__":
    main()

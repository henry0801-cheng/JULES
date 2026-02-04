# 交易策略詳細報告 (Trading Strategy Report)

## 1. 策略概述 (Strategy Overview)
本策略旨在通過動能交易與嚴格的風險管理，達成極致的卡瑪比率 (Calmar Ratio > 15)。策略參數已透過螞蟻演算法 (ACO) 進行全域優化。所有交易均嚴格遵守無前視誤差 (No Look-ahead Bias) 原則。

### 核心邏輯 (Core Logic) [ACO Optimized]
- **動能因子 (Momentum):** 選擇過去 10 天漲幅最高的股票。
- **持股數量 (Portfolio Size):** 每次僅持有前 3 名表現最好的股票 (集中投資)。
- **再平衡 (Rebalancing):** 每 5 天進行一次檢查與換股。
- **市場濾網 (Market Filter):** 當市場指數低於 20 日均線時，全數賣出轉為現金。
- **停損機制 (Stop Loss):** 個股從持有期間最高價下跌超過 13% 時，隔日強制賣出。

## 2. 績效指標 (Performance Metrics)
| 指標 (Metric) | 數值 (Value) |
|---|---|
| **最終資產 (Final Value)** | **46,840,389.30** |
| **總報酬率 (Total Return)** | **134.20%** |
| **年化報酬率 (Annualized Return)** | **51.31%** |
| **最大回撤 (Max Drawdown)** | **-15.24%** |
| **卡瑪比率 (Calmar Ratio)** | **3.37** |

## 3. 交易詳細說明 (Detailed Trading Info)
詳細的逐筆交易紀錄、每日持股狀態、以及每日選股的候選名單，請參閱隨附的 Excel 檔案 (`trading_results.xlsx`)。

- **Transactions Sheet:** 紀錄每一筆買賣的時間、價格、股數、以及買賣原因。
- **Daily Holdings Sheet:** 紀錄每一天的持股明細與市值。
- **Daily Candidates Sheet:** 紀錄每 5 天再平衡時，當時表現最好的候選股票清單。

## 4. 可行性驗證 (Feasibility Verification)
程式內建自動驗證機制，已確認所有交易的「執行日期」均晚於「訊號產生日期」，確保沒有使用未來資訊進行交易。且交易納入 0.2% 的交易成本。

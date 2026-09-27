# 02 · 訓練方法（Training Pipeline）

> 本文件對應程式碼：`mtp/dataset.py`、`mtp/cli/train.py`、`mtp/config.py`

## 1. 資料流程總覽

```
raw OHLCV parquet（每檔標的一個檔案）
        │
        ├─① 計算 16 個技術特徵  (mtp/features.add_features)
        ├─② 計算 forward label  (mtp/features.add_targets)
        │      target_h = log(close[t+h] / close[t]),  h ∈ {1,3,5,10}
        ├─③ dropna：只丟 240 天均線暖機列。尾端 label 為 NaN 的列**保留**，
        │          這樣 dataset / backtest / evaluate 的列索引才會一致
        ├─④ 全域日曆切分：所有標的共用一組切分日（pooled dates 的
        │      70% / 85% 分位 → train / val / test），每個邊界兩側各留
        │      10 個交易日 embargo
        ├─⑤ 視窗內 z-score：只用該視窗自己的 mean/std（mtp/features.standardize_windows）
        └─⑥ 滑動視窗：輸入 t-249…t，標籤取 t+1 那天的四維報酬向量
```

三個入口（`MTPStockDataset`、`mtp/cli/backtest.py`、`mtp/cli/evaluate.py`）
都走同一個 `mtp.dataset.build_symbol_split()`，因此「哪些列屬於哪個切分」
與「視窗怎麼標準化」不可能各寫各的。

### 標籤（label）定義

模型在第 `t` 天收盘後輸入 `[t-249 … t]`，預測的是**從第 t 天收盤算起**的累計對數報酬：

```
target_h = ln( close[t+h] / close[t] ),  h = 1, 3, 5, 10
```

因為是「累計」而非「單日」報酬，四個輸出天期越長量級越大，這也是為什麼
T+10 的門檻（+4.5%）天然高於 T+5（+2.5%）。

## 2. 切分策略與資料洩漏檢討

| 機制 | 做法 | 目的 |
|---|---|---|
| **全域日曆切分** | 從所有檔案的交易日**聯集**取 70% / 85% 分位，得到 `train_end`、`val_end` 兩個切分日 | 所有標的共用同一條時間軸，A 檔的訓練 label 不會落進 B 檔的評測期 |
| **以 label 歸屬判斷切分** | 樣本用「label 覆蓋到的列」決定屬於哪一段 | 訓練 label 不會跨到 val/test 的價格 |
| **Embargo** | 每個邊界兩側各留 10 個交易日（`max(horizons)`）不取樣 | 相鄰的 10 日報酬高度自相關，留白可避免邊界兩側互相污染 |
| **視窗內標準化** | mean/std 只取該 250 天視窗本身 | 不用全期統計量，避免未來資訊；訓練／回測／推論共用同一函式 |
| **特徵暖機** | 240 天均線前的 NaN 直接丟棄 | 不用前視資料填值 |
| **切分日寫進 sidecar** | `train.py` 把 `train_end` / `val_end` 存進 `<checkpoint>.json` | `evaluate.py` / `backtest.py` 一定評同一批日期，不會各算各的 |

> ⚠️ **已知限制（公開分享時請如實揭露）**：
> 1. 視窗標準化的 mean/std 來自「該視窗的 250 天」，其中包含視窗最後一天之前的資料——
>    這在推論時可以取得，因此**不是未來資料洩漏**；而且訓練與推論用的是同一套規則。
> 2. 切分日取自目前 `data_dir` 的**資料快照**；重新下載歷史會移動邊界，
>    因此評分時必須使用與訓練同一份資料（sidecar 記錄的邊界會優先採用）。
> 3. 測試期只在 `evaluate.py` / `backtest.py` 使用，**不參與選模**；
>    選模只看 validation loss。回測門檻若在測試期上反覆調整，仍構成間接過擬合。
> 4. 標籤 stride = 1，相鄰樣本重疊 249/250 天，有效獨立樣本數遠低於列印的 sample 數，
>    IC 的 t 檢定會因此偏樂觀。
> 5. 回測期的市場體質可能與訓練期同屬一段行情，屬於**體質內測試**而非完全樣本外。

## 3. 損失函數：Huber（SmoothL1Loss）

```python
criterion = nn.SmoothL1Loss()   # beta = 1.0
```

選 Huber 而非 MSE 的原因：日報酬分布是**尖峰厚尾**的，
MSE 會被單一極端日（如財報跳空、除權息）主導，導致模型去擬合離群值。
Huber 在誤差小時是平方（精細梯度）、誤差大時是線性（梯度有界），
對極端值具魯棒性，實際收斂也更穩定。

四個 horizon 的 loss **等權重相加**（`SmoothL1Loss` 對 `[B,4]` 取平均），
等同於每個天期各佔 25% 權重。若要加重長天期，可改為自訂加權：

```python
w = torch.tensor([1.0, 1.0, 1.5, 2.0])   # 偏重 T+5 / T+10
loss = (w * (pred - target).abs()).mean()  # 或用 smooth_l1_loss(reduction='none')
```

## 4. 優化器與排程

| 項目 | 值 | 說明 |
|---|---|---|
| Optimizer | AdamW | `lr=1e-4`, `weight_decay=1e-3`（Decoupled weight decay，避免 L2 與 lr 耦合） |
| Scheduler | CosineAnnealingLR | `T_max` 預設等於 `epochs`，改 `--epochs` 會自動跟著調整（`config.t_max` 可覆寫） |
| 精度 | AMP (`torch.amp.autocast` + `GradScaler`) | FP16 混合精度，記憶體與速度約省 40%；**僅在 CUDA 上啟用**，CPU 自動降回 FP32 |
| 梯度累積 | `accum_steps=2` | 實際 effective batch = 256 × 2 × GPU數 |
| 梯度裁剪 | `grad_clip=1.0` | 限制梯度範數，避免偶發爆梯度；設 0 關閉 |
| Early stop | `patience=5` | validation 連續 5 個 epoch 無改善就停；設 0 關閉 |
| 多卡 | `nn.DataParallel` | 自動偵測，單卡亦可執行 |
| 視窗長度 | `seq_len=250` | 一個交易年 |
| Epochs | 15 | 搭配 cosine 降速 |

> 訓練時 `dropout=0.2`，**推論時必須為 0**（`mtp-predict` 與 `mtp-backtest` 已固定 `dropout=0.0`）。

## 5. 執行方式

### 5.1 安裝（只需一次）

```bash
pip install -e .            # 安裝套件 + 五個指令：mtp-download / mtp-train /
                            # mtp-evaluate / mtp-predict / mtp-backtest
pip install -e ".[test]"    # 連同 pytest（測試套件）
```

`pip install -e .` 是可編輯安裝：程式碼改了立即生效，不用重裝。

不想安裝也可以跑，三種寫法執行的是同一份程式：

```bash
mtp-train --help                      # 安裝後（需在 PATH）
python -m mtp.cli.train --help        # 不用 PATH
python scripts/train.py --help        # 相容入口，不用安裝（三者同一份程式）
```

> 若 `mtp-train` 找不到，代表 pip 的使用者 Scripts 目錄不在 `PATH`。
> 用 `python -c "import site; print(site.USER_BASE)"` 找到使用者基準目錄，
> 再加上 `\Python<ver>\Scripts`，把該目錄加進使用者 `PATH`；
> 或直接改用上面後兩種寫法。

### 5.2 指令對照

| 指令 | 功能 | 等同 |
|---|---|---|
| `mtp-download` | yfinance 下載 OHLCV → parquet | `python scripts/download_data.py` |
| `mtp-train` | 訓練、存 checkpoint 與 sidecar | `python scripts/train.py` |
| `mtp-evaluate` | 測試期 skill 報表（IC vs baseline） | `python scripts/evaluate.py` |
| `mtp-predict` | 單檔即時預測 | `python scripts/predict.py` |
| `mtp-backtest` | 含成本的走訪式回測 | `python scripts/backtest.py` |

### 5.3 常用指令

```bash
# 1) 下載資料（54 檔美股核心標的 + ETF）
mtp-download --universe large

# 2) 訓練（預設 768-d / 10 層 / 15 epochs，選模只看 validation）
mtp-train

# 3) 小型化設定（單卡 8GB）
mtp-train --d-model 384 --layers 8 --batch-size 64

# 4) 指定資料夾與輸出
mtp-train --data-dir ./data/market_large --checkpoint ./checkpoints/mtp_small.pth

# 5) 評測「測試期」的預測力（IC、方向準確率、與 zero/momentum baseline 比較）
mtp-evaluate
```

### 5.4 GPU

`mtp-train` 在 CUDA 可見時**自動**用 GPU（AMP autocast + tf32 + 記憶體
pinned memory），看不到卡就退回 CPU，程式碼不需任何旗標切換。

```bash
# 1) 確認顯示卡驅動支援的 CUDA 版本（nvidia-smi 右上角 "CUDA Version"）
nvidia-smi

# 2) 裝 CUDA 版 torch（原本預設裝的是 CPU 版，只會看到 +cpu）
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 3) 驗證
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

選 wheel 的對照：`cu126` / `cu128` / `cu130` 是 CUDA runtime 版本，選**大於
等於**顯示卡需要的版本即可 —— RTX 50 系（Blackwell）至少要 `cu128`，RTX 30/40
系 `cu126` 以上即可，`cu130` 舊卡也相容（本機實測 RTX 5060 Ti + 4060 Ti 用
`2.14.0+cu130` 兩張都抓得到）。純 CPU 環境則用
`pip install torch --index-url https://download.pytorch.org/whl/cpu`。

```powershell
# 只用其中一張卡（PowerShell）
$env:CUDA_VISIBLE_DEVICES = '0'; mtp-train
# 一般 shell
CUDA_VISIBLE_DEVICES=0 mtp-train
```

- 不設定 `CUDA_VISIBLE_DEVICES` 時，`mtp-train` 會用 `nn.DataParallel`
  跨全部可見的卡（梯度累積的 effective batch 乘上卡數）；小模型／小批次時
  撿放資料的開銷可能讓多卡反而變慢，**單卡通常最快**。
- 訓練輸出的第一行會印 `GPU 0: <卡名>`；印不出來就是退回 CPU 了。
- 確認真的在跑 GPU：工作管理員的 GPU 3D/Compute 會有佔用，或看
  `Epoch 1 | ...` 每個 epoch 是否是秒級到十秒級。

其他常用參數：`--seed`（預設 42，種子涵蓋 Python / NumPy / Torch 與
data-loader shuffle）、`--patience`（validation 無改善就早停，預設 5，
0 關閉）、`--grad-clip`（梯度裁剪，預設 1.0，0 關閉）、`--train-ratio` /
`--val-ratio`（切分比例，預設 0.7 / 0.15，剩下的 15% 是測試期）。

權重存檔採「**validation loss 最佳時存檔**」，並在 `patience` 個 epoch
無改善時早停；測試期不參與任何選模。

### 訓練時螢幕輸出範例（數值依資料範圍而異）

```
GPU 0: NVIDIA GeForce RTX 3090
Calendar split (pooled over all symbols): train <= 2023-06-08 < val <= 2024-06-14 < test
[TRAIN] seq_len=250 | features=16 | targets=4 | cut-offs train<=2023-06-08 val<=2024-06-14
[VAL]   seq_len=250 | features=16 | targets=4 | cut-offs train<=2023-06-08 val<=2024-06-14
Epoch  1 | train 0.01842 | val 0.01417 *
...
Saved checkpoint -> ./checkpoints/stock_mtp_transformer.pth (best val 0.01176 @ epoch 9/15)
Score the untouched test slice with: mtp-evaluate
```

### 為什麼要 train / val / test 三段？

`val` 用來挑哪個 epoch 的權重、`test` 只在最後被 `mtp-evaluate` / `mtp-backtest`
看一次。若只有兩段，「挑最佳 epoch」這個動作本身就已經在測試期上做了 15 次
選擇，測試期就不再是樣本外了。

### 5.5 自選股票、時間範圍與下載後流程

**Q：我想下載哪些股票，要寫在哪裡？**

| 改哪個檔案 | 下載指令 | 檔案放到 |
|---|---|---|
| **`mtp/cli/universe_us.txt`**（一行一個代號、`#` 開頭是註解、**不用改程式碼**） | `mtp-download --universe universe` | `./data/market_universe` |
| `mtp/cli/download_data.py` 的 `LARGE_UNIVERSE` | `mtp-download --universe large`（預設） | `./data/market_large` ← `mtp-train` 預設讀這裡 |
| 同一個檔的 `TW_UNIVERSE` | `mtp-download --universe tw` | `./data/market_tw` |

```text
# mtp/cli/universe_us.txt
AAPL  MSFT  NVDA      # 美股直接寫代號
2330.TW               # 台股要加 yfinance 後綴 .TW
```

不想動清單也可以：自己把 `.parquet` 丟進任一資料夾，只要欄位是小寫
`date, open, high, low, close, volume`（`mtp/dataio.py` 是唯一的合約）。

**Q：時間怎麼設定？**

**沒有 `--start` / `--end` 旗標**。起始日寫死在 `mtp/cli/download_data.py`
的 `UNIVERSES`，結束日永遠是「下載當天」：

| `--universe` | 起始日 | 輸出資料夾 |
|---|---|---|
| `large` | `"2010-01-01"` | `./data/market_large` |
| `universe` | `"2015-01-01"` | `./data/market_universe` |
| `tw` | `"2000-01-01"` | `./data/market_tw` |

改第三欄那個字串，或抓完用 pandas 截斷存回。兩道門檻：`fetch()` 少於
**500 列**就不存，訓練時 `min_rows=600` 以下的檔會被跳過。
train / val / test 的切分日**不用你設**，是依下載到的資料自動算
（70% / 15% / 15% 的 pooled dates + 10 天 embargo）；改時間範圍切分日就會
跟著移動，**改完要重新訓練**，舊 checkpoint 的 sidecar 記的是舊切分日。

**Q：下載完可以直接 `mtp-train` 嗎？**

可以，前提是檔案在 `./data/market_large`（預設 `data_dir`）；下載到別的
資料夾就加 `--data-dir`：

```bash
mtp-download --universe large    # 看最後一行：failed 必須是 0
mtp-train                        # -> ./checkpoints/stock_mtp_transformer.pth + .json
mtp-evaluate                     # 先看 skill 表（IC vs zero/momentum）
mtp-backtest --tax-rate 0.0003   # evaluate 贏過 zero baseline 才值得看
mtp-predict NVDA                 # 單檔即時預測
```

**常見地雷**

- `mtp-download` 只要有檔失敗就 **exit 1**，先修好再訓練：不完整的資料夾會
  讓每個切分都默默變形。
- 所有指令都在**專案根目錄**執行，`./data/...` 是相對於目前所在資料夾。
- Windows 上 DataLoader 卡住 → 加 `--num-workers 0`。
- `mtp-predict` 永遠**即時抓 yfinance 最近 2 年**，你下載的 parquet 只給
  `mtp-evaluate` / `mtp-backtest` 用。
- 剛裝好指令要**開新終端**才會拿到 `PATH`（見 5.1）。

## 6. 常見疑問

**Q1：為什麼不放 target 的統計資訊進特徵？**
會造成 label leakage。所有特徵只依賴 `t` 及之前 的 OHLCV。

**Q2：為什麼標準化不做全體 z-score？**
金融序列非平穩，全期統計量在真實推論時取不到（未來資料）。
視窗內標準化等價於「以最近一年的分布為基準」，是最誠實的做法。

**Q3：需要每天重訓嗎？**
實務上建議每季或市場體質明顯轉換時重訓；每日只重算特徵與推論即可。

**Q4：怎麼確認模型真的學到東西，而不是只在擬合雜訊？**
跑 `mtp-evaluate`。它會對測試期的每個天期報出 IC / rankIC /
方向準確率與 t 值，並和「永遠預測 0」、「延續視窗內動能」兩個 baseline 比
Huber loss。若 model 的 Huber 不比 zero 低，或 |IC| 的 t < 2，代表沒有可
檢測的預測力——這時回測績效反映的是門檻參數，不是模型。這一步要在看任何
回測數字之前完成。

**Q5：切分日會變嗎？**
會。切分日是從當下 `data_dir` 的交易日聯集取分位數，所以重新下載資料就可能
移動。`train.py` 會把當次的 `train_end` / `val_end` 寫進 `<checkpoint>.json`，
`evaluate.py` / `backtest.py` 優先採用它們，避免用另一批日期去評分。

---
上一篇 → [01 · MTP 模型架構](01-mtp-model.md) ｜ 下一篇 → [03 · 特徵工程](03-features.md)

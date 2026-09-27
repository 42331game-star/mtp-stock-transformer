# mtp-stock-transformer

**Multi-Token Prediction (MTP) Transformer for multi-horizon stock forecasting.**

One encoder, four futures: the model predicts the cumulative log-return path for
**T+1 / T+3 / T+5 / T+10** from a 250-day window of 16 technical features.
A trade only fires when the whole path agrees — short, medium and long horizon
must resonate — which acts as a built-in noise filter.

> 📖 中文深度文件（模型架構 / 訓練方法 / 特徵 / 回測）：**[docs/zh/](docs/zh/01-mtp-model.md)**

---

## Usage: pick your tickers → download → train

The four questions people ask first. (中文版：[docs/zh/02-training.md §5](docs/zh/02-training.md),
installation details: [Installation](#installation).)

### 1. Where do I write the stocks I want?

| You edit | Download with | Files land in |
|---|---|---|
| **`mtp/cli/universe_us.txt`** — one ticker per line, `#` starts a comment, *no code changes* | `mtp-download --universe universe` | `./data/market_universe` |
| `LARGE_UNIVERSE` in `mtp/cli/download_data.py` | `mtp-download --universe large` *(default)* | `./data/market_large` ← what `mtp-train` reads by default |
| `TW_UNIVERSE` in the same file | `mtp-download --universe tw` | `./data/market_tw` |

```text
# mtp/cli/universe_us.txt
AAPL  MSFT  NVDA           # US tickers, plain symbol
2330.TW                    # Taiwan: yfinance suffix is .TW
```

Prefer not to touch the lists at all? Drop your own `.parquet` files into any data
folder — the only contract is lowercase `date, open, high, low, close, volume`
(`mtp/dataio.py` validates it).

### 2. How do I set the time range?

There is **no `--start` / `--end` flag**. The start date is a literal in
`UNIVERSES` inside `mtp/cli/download_data.py`; the end date is always *today*:

| `--universe` | Start date | Output folder |
|---|---|---|
| `large` | `"2010-01-01"` | `./data/market_large` |
| `universe` | `"2015-01-01"` | `./data/market_universe` |
| `tw` | `"2000-01-01"` | `./data/market_tw` |

```python
UNIVERSES = {
    "large": (LARGE_UNIVERSE, "./data/market_large", "2010-01-01"),   # <- change this
    ...
}
```

...or download first and trim the parquet with pandas. Two gates apply: `fetch()`
discards anything shorter than **500 rows**, and training skips symbols shorter
than **600 rows** (`min_rows` in `mtp/config.py`).

You never set the train/val/test dates yourself — they are derived from whatever
you downloaded (70% / 15% / 15% of the pooled calendar, plus a 10-session
embargo), so re-downloading with a different range shifts them.

### 3. Can I run `mtp-train` right after the download?

Yes — **if** the files are in `./data/market_large`, the default `data_dir`.
For any other folder pass `--data-dir`:

```bash
mtp-download --universe large    # check the last line: "failed=0" must hold
mtp-train                        # -> ./checkpoints/stock_mtp_transformer.pth + .json
mtp-evaluate                     # skill table FIRST (IC vs zero/momentum baseline)
mtp-backtest --tax-rate 0.0003   # only worth reading if mtp-evaluate says you win
mtp-predict NVDA                 # single-name forecast
```

### 4. Common gotchas

* `mtp-download` exits **1** when any ticker failed — fix those first: a partial
  folder silently poisons every split downstream.
* Run everything from the repo root; `./data/...` is relative to your current
  directory.
* Windows hangs in the DataLoader → add `--num-workers 0`.
* `mtp-predict` always fetches the **last 2 years live from yfinance**; your
  downloaded parquet files are used by `mtp-evaluate` / `mtp-backtest` only.
* First time using the commands? Open a **new** terminal so the updated `PATH`
  takes effect (see [Installation](#installation)).

---

## Why multi-token prediction?

A single next-day-return head is dominated by noise. Predicting *several* future
steps at once forces the backbone to learn the temporal structure
(intraday momentum → multi-week trend) and gives the trading layer a **path**
to validate instead of a single number:

```
x: [B, 250, 16]  ──►  TransformerEncoder (10L / 768d / 16H)
                                   │
                     pool = mean(h) + h[-1]
                                   │
             ┌─────────┬───────────┼───────────┐
          head_d1   head_d3    head_d5     head_d10
             └─────────┴───────────┴───────────┘
                                   │
                       y: [B, 4]  =  log-return path
```

| Property | Value |
|---|---|
| Input | 250 trading days × 16 features |
| Output | cumulative log-returns at T+1 / T+3 / T+5 / T+10 |
| Backbone | pre-norm Transformer encoder, d_model 768, 10 layers |
| Pooling | `mean(seq) + last(seq)` — global context fused with the latest step |
| Heads | 4 independent 2-layer MLP heads (`768 → 128 → 1`) |
| Loss | Huber (`SmoothL1Loss`), robust to fat-tailed returns |
| Params | ≈ 86M (checkpoint ≈ 288 MB) |

## Repository layout

```
mtp-share/
├── README.md                  # this file (English)
├── pyproject.toml             # packaging: `pip install -e .` installs mtp-*
├── requirements.txt           # same dependencies, for `pip install -r`
├── LICENSE / .gitignore
├── mtp/                       # the package
│   ├── __init__.py            # public API + __version__
│   ├── config.py              # every hyper-parameter in one place
│   ├── dataio.py              # yfinance/OHLCV column normalisation contract
│   ├── features.py            # 16 shared features + window-local scaling
│   ├── dataset.py             # global calendar split, embargo, sliding windows
│   ├── checkpoint.py          # weight/sidecar/cut-off loading (shared by scripts)
│   ├── model.py               # MTPStockTransformer
│   └── cli/                   # the five entry points (installed as console scripts)
│       ├── download_data.py   # mtp-download: yfinance → parquet (US core/broad, TW50)
│       ├── train.py           # mtp-train: AMP + gradient accumulation + multi-GPU
│       ├── evaluate.py        # mtp-evaluate: test-slice skill report, IC vs baselines
│       ├── predict.py         # mtp-predict: interactive single-ticker forecast
│       ├── backtest.py        # mtp-backtest: walk-forward simulation with costs
│       └── universe_us.txt    # ticker list for --universe universe
├── scripts/                   # thin shims: `python scripts/train.py` == `mtp-train`
├── tests/                     # offline tests (no network, no market data)
│   ├── helpers.py             # synthetic OHLCV generators
│   ├── test_smoke.py          # features → dataset → model → checkpoint
│   ├── test_split.py          # no label crosses a cut-off; window-local scaling
│   ├── test_dataio.py         # yfinance column/date layouts
│   ├── test_cli.py            # every flag must map to a real config field
│   └── test_evaluate.py       # metric helpers (IC, baselines, Huber)
└── docs/
    └── zh/                    # 中文深度文件
        ├── 01-mtp-model.md    # 架構與設計理由
        ├── 02-training.md     # 訓練流程、切分、防洩漏檢討（含安裝與 GPU 用法）
        ├── 03-features.md     # 16 維特徵逐項說明
        └── 04-backtest.md     # 交易規則、成本、回測限制
```

## Installation

```bash
pip install -e .            # package + the five `mtp-*` commands
pip install -e ".[test]"    # the above plus pytest
```

Every command can be invoked three ways — they all execute the same code:

| Form | When to use |
|---|---|
| `mtp-train --help` | after installing; needs the pip `Scripts` dir on `PATH` |
| `python -m mtp.cli.train --help` | after installing; no `PATH` needed |
| `python scripts/train.py --help` | no install needed (source checkout) |

> Windows: pip's user `Scripts` folder is often **not** on `PATH`. Locate it with
> `python -c "import site; print(site.USER_BASE)"` and append `\Python<ver>\Scripts`,
> add that folder to the user `PATH`, or use the other two forms.

`requirements.txt` lists the same dependencies as `pyproject.toml` for people who
prefer `pip install -r requirements.txt`; the `pyproject.toml` copy is the one
`pip install` uses.

## How the data is split

All symbols share **one calendar**. The cut-offs are taken from the pooled
trading dates of every file in `data_dir` (default 70% train / 15% validation /
15% test) and each boundary is padded with a **10-session embargo** on both
sides, so a training label can never touch a price path that the validation or
test slice scores — even though the boundaries are global while symbol
histories differ.

* `mtp-train` selects the checkpoint on **validation** loss only.
* `mtp-evaluate` and `mtp-backtest` both score the untouched
  **test** slice and read the cut-off dates from the checkpoint sidecar, so the
  three scripts cannot drift apart.
* Every window is z-scored with **its own** mean/std (never full-series
  statistics), identically in training, back-testing and live inference.

## Quick start

```bash
pip install -e .            # once; adds mtp-download / mtp-train / mtp-evaluate /
                            #      mtp-predict / mtp-backtest

# 1) download data (54 core US names + ETFs)
mtp-download --universe large

# 2) train (15 epochs, 768-d / 10 layers) on the train/validation slices
mtp-train

# 3) skill report on the untouched test slice — do this BEFORE backtesting
mtp-evaluate

# 4) forecast a single ticker
mtp-predict NVDA

# 5) walk-forward backtest on the same test slice
mtp-backtest                                # Taiwan costs (default: 0.05% + 0.3% sell tax)
mtp-backtest --tax-rate 0.0003              # US costs (0.05% + 0.03%)

# 6) run the test-suite (offline, synthetic data only)
python -m pytest tests -q
```

Every command also works as `python -m mtp.cli.<module>` or
`python scripts/<module>.py` (see [Installation](#installation)).

`mtp-evaluate` reports per-horizon IC / rank-IC / directional accuracy
against a **zero** and a **momentum** baseline. If it says the model loses to
`zero`, no backtest number is meaningful — see [docs/zh/04-backtest.md §5](docs/zh/04-backtest.md).

Smaller GPU? Use a lighter configuration:

```bash
mtp-train --d-model 384 --layers 8 --batch-size 64
```

Reproducibility: `--seed` (default 42) seeds Python / NumPy / Torch and the
data-loader shuffles; `--patience` early-stops on validation loss; `--grad-clip`
is on by default.

## GPU

`mtp-train` uses CUDA automatically when it can see a GPU (AMP autocast + tf32
+ pinned memory) and falls back to CPU otherwise — no flag switches devices.

```bash
nvidia-smi                      # check the driver's "CUDA Version"
pip install torch --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

* The default `pip install torch` is a **CPU-only** build (`2.x.y+cpu`) and
  `torch.cuda.is_available()` returns `False` — you must install a `+cuXXX`
  wheel from `https://download.pytorch.org/whl/`.
* Pick a wheel whose CUDA runtime is **≥ what the card needs**:
  `cu126` for RTX 20/30/40, `cu128` minimum for RTX 50 (Blackwell), `cu130`
  works across them. Example on a box with an RTX 5060 Ti + RTX 4060 Ti:
  `torch 2.14.0+cu130`, both cards visible.
* Without `CUDA_VISIBLE_DEVICES` the trainer wraps the model in `nn.DataParallel`
  over every visible card; with a small model the scatter/gather overhead can
  make that *slower* than one card. Pin a single GPU when in doubt:

  ```bash
  CUDA_VISIBLE_DEVICES=0 mtp-train                 # bash / zsh
  $env:CUDA_VISIBLE_DEVICES = '0'; mtp-train       # PowerShell
  ```

* The first training line prints `GPU 0: <device name>`; if it is missing you
  are on CPU. On CPU, `--num-workers 0` avoids Windows worker-process overhead.
* Full notes: [docs/zh/02-training.md §5.4](docs/zh/02-training.md).

## Trading rules (summary)

| Rule | Value |
|---|---|
| Entry | `T+5 ≥ +2.5%` **and** `T+10 ≥ +4.5%` **and** `T+3 > −1%` |
| Cost guard | `T+10 ≥ 5 × round-trip cost` (≈ 2.0% in Taiwan) |
| Crash guard | `T+1 > −15%` (skip obvious collapse / broken data) |
| Stop | trailing `2.3 × ATR(14)` below the running high |
| Time exit | 10 sessions (`extension_thresh` exists in the config but the 30-day extension rule is **not implemented**) |
| Costs | 0.05% commission/side + 0.3% sell tax (TW), or 0.03% (US) |

The thresholds live in `mtp/config.py` and are the same values
`mtp-predict` uses to label a forecast an "entry path" — one source of
truth. Full rationale in [docs/zh/04-backtest.md](docs/zh/04-backtest.md).

## Data

| Universe | Folder | Size |
|---|---|---|
| US core (training) | `data/market_large` | 54 tickers/ETFs |
| US broad pool | `data/market_universe` | 200+ tickers |
| Taiwan 50 | `data/market_tw` | 50 tickers |

`data/`, `checkpoints/` and all `*.parquet` / `*.pth` are git-ignored —
regenerate them with `mtp-download` and `mtp-train`.

> `mtp-train` also writes a `<checkpoint>.json` sidecar recording the exact
> architecture (d_model, layers, horizons…) **and the calendar cut-off dates**,
> so `mtp-predict` / `mtp-evaluate` / `mtp-backtest` can reload a non-default
> configuration and score exactly the slice that was held out.

## Known limitations

Read this before trusting any number in a backtest:

1. **Prove skill first.** `mtp-evaluate` must show the model beating the
   zero baseline with an IC that is statistically meaningful (`|t| ≥ 2`). If it
   does not, the backtest is measuring the hand-tuned thresholds, not the model.
2. Thresholds (2.5% / 4.5% / 2.3×ATR) were chosen on historical data → risk of
   indirect overfitting. Sensitivity checks are still on the to-do list.
3. Fill assumptions are optimistic: stops fill exactly at the stop line, there
   is no slippage or liquidity model, and the simulation updates the trailing
   line with the day's high before testing the day's low (intra-bar ordering is
   ambiguous in both directions).
4. Entries are priced at the **signal day's close**, which is only known after
   that close — treat it as a lower bound on achievable execution.
5. The portfolio curve is an approximation: mean net return per entry date
   divided by average holding days, compounded. It ignores overlapping
   positions, capital limits and position sizing (no 5-slot model is coded).
6. Windows stride by one day, so adjacent samples share 249/250 days. Sample
   counts and IC t-statistics therefore overstate the effective sample size.
7. The universe is a curated list of surviving large caps and ETFs, and the
   default test slice is bull-heavy. Survivorship and regime risk both apply.
8. Cut-off dates are derived from the data snapshot. Re-downloading history
   shifts them, so score a checkpoint against the snapshot it was trained on.
9. Educational / research use only — **not investment advice**.

## License

MIT (see `LICENSE`).

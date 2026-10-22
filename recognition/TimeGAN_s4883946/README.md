# TimeGAN for Limit Order Book Simulation (LOBSTER AMZN, Level 10)

Project 2.7 (Time-Series Generation: Limit Order Book and Market Simulation), Hard difficulty.

Author: Jamie Landolt (s4883946)

## Overview

**Problem.** Evaluating trading algorithms and risk systems needs realistic market simulations, but real limit order book (LOB) data is expensive. A generative model can produce synthetic order book sequences, but an order book is not an ordinary time series: it must obey hard rules. The best bid must be below the best ask (a *crossed book* would be a free arbitrage), prices must be ordered across the 10 depth levels, volumes cannot be negative, and the sequence should show real stylised facts such as volatility clustering and heavy-tailed returns. This project trains a **TimeGAN** on one day of AMZN Level-10 data from LOBSTER and audits whether the generated 24-second order book windows respect those rules and match the real distributions, compared with a plain recurrent GAN baseline.

**How it works.** Each real window is a 24-step sequence (one step per second) of 41 features: the mid-price log-return, the distance of each of 10 ask and 10 bid levels from the mid-price, and the volume at each of those 20 levels. TimeGAN (Yoon, Jarrett and van der Schaar, NeurIPS 2019) learns in a *latent space*: an embedder and recovery network (an autoencoder) map windows to and from latent sequences, a generator turns random noise into latent sequences, a supervisor learns the step-to-step dynamics of the latent sequences, and a discriminator tells real latents from generated ones. It is trained in three phases (autoencoder, supervisor, then everything jointly), using reconstruction, supervised, adversarial and mean/std-matching losses. The baseline is a GRU generator and discriminator that work directly on the features with only the adversarial loss. Generated windows are decoded back to prices and volumes and audited for crossed books, invalid price ladders and negative depth, then compared with real held-out windows using KL divergence, SSIM of depth heatmaps and volatility clustering.

```
real window x --Embedder--> h --Recovery--> x~          (reconstruction loss)
                            |
                            +--Supervisor--> next-step prediction of h  (supervised loss)

noise z --Generator--> e^ --Supervisor--> h^ --Recovery--> x^   (synthetic window)

Discriminator: real latents h   vs   generated latents h^, e^    (adversarial loss)
```

## Files

| File | Purpose |
|---|---|
| `modules.py` | TimeGAN networks and the recurrent-GAN baseline (PyTorch only) |
| `dataset.py` | Loading, 1-second resampling, features, return transforms, chronological split, windows, data loaders |
| `train.py` | Training, validation, test and checkpointing for both models; loss curves |
| `predict.py` | Loads trained weights, samples windows, compares with real data, saves figures |
| `metrics.py` | Microstructure audit, KL divergence, volatility clustering, depth-heatmap SSIM |
| `test_pipeline.py` | Unit and end-to-end tests (use a small fake order book, no real data needed) |
| `audit_data.py`, `plot_returns.py` | Real-data audit and return-distribution figure |
| `slurm_train.sh`, `slurm_pack.sh`, `run_experiments.sh`, `summarise_runs.py` | Running and summarising experiments on the Rangpur cluster |

## Dependencies and reproducibility

Results were produced on the Rangpur cluster (NVIDIA A100 GPUs) with these exact versions:

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.1 |
| scikit-image | 0.26.0 |

`modules.py` depends only on PyTorch. The LOBSTER sample files are **not** committed (assessment rule); place the two CSV files and the LOBSTER readme in one directory and pass it with `--data-dir`. Every run takes `--seed`; the reported numbers are the mean and standard deviation over seeds 0, 1 and 2. GPU training is not bit-for-bit deterministic, so single runs can differ slightly.

## Usage

```bash
# unit and end-to-end tests (no real data needed)
python -m unittest test_pipeline -v

# audit of the real data, and the return-distribution figure
python audit_data.py <data_dir>
python plot_returns.py <data_dir> figures/return_distribution.png

# train (outputs/<model>.pt, <model>_summary.json, <model>_training.png)
python train.py --model timegan  --data-dir <data_dir> --out-dir outputs
python train.py --model baseline --data-dir <data_dir> --out-dir outputs
# optional TimeGAN changes studied below:
#   --return-transform quantile --ladder increments --feature-adv --w-diff 10 --step 1

# sample from the trained models, print the comparison table and save figures
python predict.py --data-dir <data_dir> --model-dir outputs

# on Rangpur (Slurm): one run, or a whole experiment group
sbatch --job-name=lob_timegan slurm_train.sh timegan
./run_experiments.sh transform        # also: step, loss, depth, ladder, combined
python summarise_runs.py outputs/transform
```

## Feasibility Review

**Audience:** project manager (tutor). **Question under review:** can a TimeGAN produce synthetic order books that are safe to use for backtesting, i.e. that respect hard microstructure invariants and not just look plausible?

### 1. User need, scope and acceptance criteria

The intended user is a quant developer who needs realistic order-book simulations to stress-test execution strategies without paying for tick-data licences. The prototype generates windows of 10-level order book snapshots and audits them against microstructure rules. Acceptance criteria (all measured on a held-out, chronologically later test period, against the baseline on the same split):

1. **Structural validity:** crossed or locked books (P_bid,1 >= P_ask,1) in generated data are reported as a rate per 1,000 snapshots, with the real data's rate (0, see below) as the reference. Target: 0 for the final model; the baseline's rate is reported for comparison.
2. **Distributional fidelity:** KL divergence between real and synthetic distributions of spread and of mid-price returns <= 0.1 (spec target). Real data from the training period already scores about 0.2 against the test period (section 3), so the achievable floor is reported next to every result.
3. **Depth structure:** SSIM > 0.6 between real and synthetic depth heatmaps (price level x time step) (spec target). Held-out real windows only reach about 0.28 against their nearest real window, so results are reported against that ceiling as well as against the target.
4. **Stylised facts:** the autocorrelation of absolute returns inside 24-step windows at lags 1, 5 and 10 is positive and decays as in the real test data (0.14 / 0.08 / 0.005), rather than collapsing to roughly 0 (white noise) or being far too persistent.
5. **Resource constraint:** peak GPU memory <= 2 GB and training time <= 30 minutes per model on one GPU (first full runs measured 0.57 GB and 9 minutes for TimeGAN).

### 2. Model choice and course concepts

- **Dataset:** LOBSTER sample, AMZN, 2012-06-21, 09:30 to 16:00, 10 levels, 269,748 events. Prices are in units of 1/10,000 dollar.
- **Model:** TimeGAN (Yoon, Jarrett and van der Schaar, NeurIPS 2019): an embedding/recovery autoencoder, a supervised next-step loss in latent space, and an adversarial loss, which together are meant to capture temporal dynamics better than a plain GAN. **Difficulty band:** Hard.
- **Baseline:** a vanilla recurrent GAN (GRU generator and discriminator) with only the adversarial loss, so the comparison isolates what TimeGAN's supervised and reconstruction losses add.
- **Preprocessing:** log-returns for prices, volume normalisation, resampling onto a fixed time grid, and a chronological train/validation/test split (no shuffling across time, to avoid look-ahead bias).
- **Course concepts:** GAN training and its instabilities (mode collapse, balancing generator and discriminator), recurrent sequence models, and autoencoders. Literature support is the TimeGAN paper above and the LOBSTER data description (Huang and Polak, 2011).

### 3. Preliminary feasibility evidence (data audit)

`audit_data.py` was run on the real sample on Rangpur. Findings:

| Check | Result |
|---|---|
| Crossed or locked books | 0 of 269,748 |
| Non-monotonic bid or ask ladders | 0 |
| Negative or zero depth | 0 |
| Spread | $0.01 to $0.77 (median $0.13) |
| Zero mid-price change per event / per 1 s grid step | 89.8% / 70.1% |
| Return kurtosis per event / per 1 s grid step | about 203 / 38.5 (Gaussian = 3) |
| Autocorrelation of absolute returns, per event (lags 1 / 5 / 20) | 0.25 / 0.15 / 0.10 |

![Real 1-second mid-return distribution: spike at zero and fat tails](figures/return_distribution.png)

Implications: (a) the real data is clean, so any invariant violation in synthetic data is attributable to the model; (b) volatility clustering is present and can be tested; (c) returns have a sharp spike at zero and very fat tails (figure), so I model a 1-second grid rather than raw events; (d) volumes are heavily skewed (median 100, max 5,700 shares), so they use log scaling.

**Early results from the pipeline and first full runs** (one seed, trained on Rangpur): TimeGAN has 480k parameters, trained in about 9 minutes with 0.57 GB peak GPU memory; the baseline GAN took about 2 minutes. Neither generator produced a crossed book, but the baseline broke the price-ladder ordering in 98.6% of snapshots against 14.6% for TimeGAN, so the central question is measurable. Neither is close to the distribution targets (KL about 1.3 for spread and returns, against about 0.2 for real train-vs-test data). Real windows reach only about 0.28 SSIM against their nearest real neighbour, so a perfect generator would not reach the 0.6 target under this definition.

### 4. Risks, budget and fallback

| Risk | Mitigation |
|---|---|
| The sample is a single trading day, so the "later hours" test set is small and the result may not generalise across days | State this limitation explicitly; use a chronological split and report uncertainty rather than claiming general market validity |
| Return KL <= 0.1 may be very hard: real returns have a sharp spike at zero (70% zeros) and fat tails (kurtosis 38.5), while a generator fed Gaussian noise and trained with MSE and moment-matching losses tends towards a smooth bell-shaped hump, which the first run is consistent with (hypothesis, not yet tested). z-scoring with a few extreme outliers may also let those outliers dominate the losses. KL is also noisy in the sparse tails and depends on binning | Fix the binning rule up front and report it; run a normalisation ablation (plain z-score vs clipped vs quantile transform, and modelling exact zeros separately) and compare each against the real train-vs-test KL floor of about 0.2 |
| The spec's KL <= 0.1 and SSIM > 0.6 targets may be unattainable on a single day of data (real train-vs-test KL about 0.2; real-vs-nearest-real SSIM about 0.28) | Report every result next to these real-data references, so it is clear how much of any gap comes from the single-day data and how much from the model |
| TimeGAN training instability or mode collapse | Train the baseline first, monitor losses and sample diversity, tune the learning rate and the balance of the three losses |
| Raw TimeGAN outputs are unconstrained and may cross the book | Report violation rates on raw outputs as the main result; any decoding constraint (for example enforcing positive spread by construction) is reported separately, because it would not show the model learning the invariant |
| Rangpur GPU queue time | Short smoke-test jobs first, then one longer batch job |

**Budget:** one GPU on the `comp3710` partition via Slurm. A full TimeGAN run takes about 9 minutes, so there is room for repeated runs and several seeds.
**Next experiment:** the normalisation ablation above, measured on the same chronological validation and test splits, plus repeat runs with different seeds so differences between models are not single-run noise.
**Fallback:** if TimeGAN will not train stably, report the baseline plus a TimeGAN with a simplified architecture and shorter windows, and document the failure modes honestly, since the project question is whether the model enforces the invariants and not only whether it succeeds.

## Data and preprocessing

**Data.** LOBSTER sample file for AMZN on 2012-06-21, 09:30 to 16:00, 10 price levels: 269,748 order book events (Huang and Polak, 2011). Prices are stored in units of 1/10,000 dollar, and the minimum price step (tick) is $0.01.

**Preprocessing** (all in `dataset.py`):

1. **Resample onto a 1-second grid.** Events arrive at irregular times and about 90% of them leave the mid-price unchanged. For each second, the book is the state after the last event at or before that time. This gives 23,400 evenly spaced snapshots; 70% of 1-second mid-returns are still exactly zero (event level: 90%), and the kurtosis falls from about 203 to 38.5.
2. **Features (41 per step).** The mid-price log-return `ln(P_t / P_{t-1})`; the distance of each of the 10 ask and 10 bid levels from the mid-price in ticks, as `log1p(distance)`; and `log1p(volume)` for each of the 20 levels (volumes are very skewed: median 100, maximum 5,700 shares). Expressing prices as distances from the mid-price turns the market rules into sign and ordering conditions that can be audited, instead of building them into the representation.
3. **Optional return transform** (`--return-transform`): plain z-scoring, winsorising at the 0.5th and 99.5th percentiles (`clip`), or a quantile transform (`quantile`) that maps returns to normal scores through the training-set empirical CDF. In the quantile transform all exact zeros map to one value and decoding sends a whole band of normal scores back to exactly 0, so the generator does not have to output exact zeros.
4. **Optional ladder representation** (`--ladder`): either the distance of every level from the mid-price (`levels`), or the distance of level 1 followed by the gap between successive levels (`increments`). Real gaps are always at least one tick, so with `increments` the model only has to produce positive gaps for the ladder to stay ordered. Nothing is enforced: a negative decoded gap is counted as a violation.
5. **Chronological split, then normalisation.** Each feature is z-scored with the mean and standard deviation of the **training split only**.
6. **Windows.** Overlapping 24-step windows (stride 1), built inside each split.

**Splits and why.** The day is split in time order, with no shuffling: the first 60% for training (09:30 to 13:24), the next 20% for validation (13:24 to 14:42) and the last 20% for testing (14:42 to 16:00). Models only see earlier data than they are validated and tested on, which prevents look-ahead bias. Windows never cross a split boundary, and normalisation statistics and the quantile transform are fitted on the training split only, so nothing leaks. This gives 14,017 training, 4,657 validation and 4,657 test windows. Model selection (the best checkpoint) uses the validation split; the test split is only used for the reported numbers. Because the sample is a single day, the test period is a different part of the same day and not a different day, so results should not be read as evidence about other days or market conditions.

![Real 1-second mid-return distribution: spike at zero and fat tails](figures/return_distribution.png)

*Real 1-second mid-returns have a sharp spike at zero (70% exact zeros) and far heavier tails than a Gaussian of the same standard deviation (kurtosis 38.5 against 3).*

## Evaluation

All metrics compare generated windows with **real test windows** (`metrics.py`).

- **Microstructure audit** (per snapshot): *crossed* (best ask not above best bid, i.e. spread <= 0), *ladder* (ask prices not rising, or bid prices not falling, with depth level), *negative depth*.
- **KL divergence** between real and generated spread, and between real and generated 1-step mid-returns. Binning rule (fixed so results are comparable): 50 equal-width bins between the 0.5th and 99.5th percentile of the real data plus two open-ended outer bins, with a small constant added to each bin before normalising.
- **Volatility clustering:** autocorrelation of absolute returns inside windows at lags 1 and 5.
- **SSIM of depth heatmaps.** The heatmap of a window is a 20 x 24 image: rows are the bid levels 10 to 1 followed by the ask levels 1 to 10, columns are the 24 time steps, and the colour is `log(1 + volume)`. Each generated heatmap is compared with its nearest real heatmap (smallest L2 distance), and the SSIM is averaged over 300 generated windows. This is a strict measure, so each table also shows the **real-data reference**: the same score for real training windows against real test windows, which is the highest a perfect generator could reasonably reach.

**Real-data reference.** The test period differs from the training period (the average spread is $0.14 in training and $0.11 in the test period), so even real data from the training period does not match the test period perfectly: at the 1-second step its KL is about 0.22 for spread and 0.19 for returns, and its SSIM is 0.30. Targets taken from the assignment (KL <= 0.1, SSIM > 0.6) should be read with these numbers in mind.

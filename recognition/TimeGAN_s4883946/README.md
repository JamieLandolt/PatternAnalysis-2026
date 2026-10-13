# TimeGAN for Limit Order Book Simulation (LOBSTER AMZN, Level 10)

Project 2.7 (Time-Series Generation: Limit Order Book and Market Simulation), Hard difficulty.

Author: Jamie Landolt (s4883946)

Work in progress: the sections below are filled in as the project develops.

## Feasibility Review

**Audience:** project manager (tutor). **Question under review:** can a TimeGAN produce synthetic order books that are safe to use for backtesting, i.e. that respect hard microstructure invariants and not just look plausible?

### 1. User need, scope and acceptance criteria

The intended user is a quant developer who needs realistic order-book simulations to stress-test execution strategies without paying for tick-data licences. The prototype generates windows of 10-level order book snapshots and audits them against microstructure rules. Acceptance criteria (all measured on a held-out, chronologically later test period, against the baseline on the same split):

1. **Structural validity:** crossed or locked books (P_bid,1 >= P_ask,1) in generated data are reported as a rate per 1,000 snapshots, with the real data's rate (0, see below) as the reference. Target: 0 for the final model; the baseline's rate is reported for comparison.
2. **Distributional fidelity:** KL divergence between real and synthetic distributions of spread and of mid-price returns <= 0.1 (spec target). Real data from the training period already scores about 0.2 against the test period (section 3), so the achievable floor is reported next to every result.
3. **Depth structure:** SSIM > 0.6 between real and synthetic depth heatmaps (price level x time step) (spec target). Held-out real windows only reach about 0.28 against their nearest real window, so results are reported against that ceiling and I will ask staff to negotiate the target, which the spec allows.
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

**Early results from the pipeline and first full runs** (one seed, trained on Rangpur): TimeGAN has 480k parameters, trained in about 9 minutes with 0.57 GB peak GPU memory; the baseline GAN took about 2 minutes. Neither generator produced a crossed book, but the baseline broke the price-ladder ordering in 98.6% of snapshots against 14.6% for TimeGAN, so the central question is measurable. Neither is close to the distribution targets (KL about 1.3 for spread and returns, against about 0.2 for real train-vs-test data). Real windows reach only about 0.28 SSIM against their nearest real neighbour, which is why the 0.6 target needs negotiating.

### 4. Risks, budget and fallback

| Risk | Mitigation |
|---|---|
| The sample is a single trading day, so the "later hours" test set is small and the result may not generalise across days | State this limitation explicitly; use a chronological split and report uncertainty rather than claiming general market validity |
| Return KL <= 0.1 may be very hard: real returns have a sharp spike at zero (70% zeros) and fat tails (kurtosis 38.5), while a generator fed Gaussian noise and trained with MSE and moment-matching losses tends towards a smooth bell-shaped hump, which the first run is consistent with (hypothesis, not yet tested). z-scoring with a few extreme outliers may also let those outliers dominate the losses. KL is also noisy in the sparse tails and depends on binning | Fix the binning rule up front and report it; run a normalisation ablation (plain z-score vs clipped vs quantile transform, and modelling exact zeros separately) and compare each against the real train-vs-test KL floor of about 0.2 |
| The spec's KL <= 0.1 and SSIM > 0.6 targets may be unattainable on a single day of data (real train-vs-test KL about 0.2; real-vs-nearest-real SSIM about 0.28) | Report results against these real-data references and ask staff to agree adjusted targets |
| TimeGAN training instability or mode collapse | Train the baseline first, monitor losses and sample diversity, tune the learning rate and the balance of the three losses |
| Raw TimeGAN outputs are unconstrained and may cross the book | Report violation rates on raw outputs as the main result; any decoding constraint (for example enforcing positive spread by construction) is reported separately, because it would not show the model learning the invariant |
| Rangpur GPU queue time | Short smoke-test jobs first, then one longer batch job |

**Budget:** one GPU on the `comp3710` partition via Slurm. A full TimeGAN run takes about 9 minutes, so there is room for repeated runs and several seeds.
**Next experiment:** the normalisation ablation above, measured on the same chronological validation and test splits, plus repeat runs with different seeds so differences between models are not single-run noise.
**Fallback:** if TimeGAN will not train stably, report the baseline plus a TimeGAN with a simplified architecture and shorter windows, and document the failure modes honestly, since the project question is whether the model enforces the invariants and not only whether it succeeds.

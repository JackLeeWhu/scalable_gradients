# Reproduction results: Sections 5 and 7

**Jingxiang Xu.** Independent study of [Li et al. (2020)](https://proceedings.mlr.press/v108/li20i.html) and its [supplement](https://proceedings.mlr.press/v108/li20i/li20i-supp.pdf). All measurements below come from the code and output files in this directory.

## 1. Model and experimental scope

Section 5 introduces a latent SDE with a prior drift, a data-conditioned posterior drift, and a shared diffusion. Training minimizes the negative evidence lower bound: observation negative log-likelihood plus initial-state and pathwise KL terms. In our code, `torchsde.sdeint_adjoint(..., logqp=True)` integrates the posterior path, estimates its pathwise KL contribution, and computes gradients with the stochastic adjoint. Figure 3 is a model diagram, not a separate numerical experiment.

Section 7 contains three tasks: numerical gradient checks (7.1), synthetic latent SDE examples (7.2), and CMU motion forecasting (7.3). The published random draws, trained weights, and several implementation settings are unavailable, so the numerical figures below are independent replications of the specified experiments rather than copies of the authors' data points.

## 2. Numerical gradients (Section 7.1)

For Supplementary Example 2, we compare direct differentiation through a Milstein solver and a Milstein stochastic adjoint against an analytic gradient. Each setting uses 10 parameters, 16 trajectories per draw, and 64 independent Brownian seeds. Both methods use the same paths within each comparison.

| Milstein steps | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|
| Direct backpropagation, median gradient MSE | 4.08×10⁻⁵ | 1.04×10⁻⁵ | 2.60×10⁻⁶ | 5.67×10⁻⁷ |
| Stochastic adjoint, median gradient MSE | 1.89×10⁻⁶ | 4.27×10⁻⁷ | 1.19×10⁻⁷ | 2.50×10⁻⁸ |

The error decreases as the step size shrinks. Adaptive SRK forward integration with an adaptive Milstein adjoint gives median total forward-plus-backward function evaluations of **494, 1,033, and 3,898** at absolute tolerances 10⁻², 10⁻³, and 10⁻⁴, respectively; corresponding median gradient RMSE is **0.00123, 0.000285, and 0.0000618**. On this CPU and short test problem, the median time/error pairs were Euler backpropagation (256 steps) **0.084 s / 0.00491**, Milstein backpropagation (64 steps) **0.0287 s / 0.00138**, and Milstein adjoint (64 steps) **0.120 s / 0.000284**. These timings describe this hardware and implementation; they do not establish the paper's large-scale memory or speed claims.

[Figure 4-style fixed-step result](outputs/numerical_study.png) · [adaptive result](outputs/adaptive_study.png) · [CPU result](outputs/efficiency_study.png) · [Supplementary Figure 7-style panels](outputs/supplement_figure7_style.png). The underlying [CSV](outputs/numerical_study.csv) and [summary](outputs/numerical_study_summary.json) are included.

**Printed-formula corrections.** The drift printed for Supplementary Example 2 is inconsistent with its stated exact solution: applying Itô's formula to the latter gives `−p² sin(x) cos³(x)`, which is the drift used here. Example 1 also swaps the roles of its two parameters between the printed SDE and exact solution; our six-panel study uses the exact solution of the printed SDE. The paper does not specify every loss, seed, or solver option needed for pointwise numerical agreement.

## 3. Synthetic latent SDEs (Section 7.2)

We generated 1,024 GBM paths on 51 observation times and 1,024 stochastic Lorenz paths on 41 times, following the supplement's parameters and adding observation noise with standard deviation 0.01. Lorenz uses an Euler–Maruyama internal step of 0.005 because the paper does not state its generation step. Models have four-dimensional latent states, a reverse GRU encoder, 100-unit single-hidden-layer drift/diffusion networks, and a linear decoder. We used a one-dimensional encoder context, Adam with initial learning rate 0.01 and 0.999 per-step decay, stochastic adjoint differentiation, a 0.01 training solver step, 300 iterations, batch size 64, and a 50-iteration linear KL warmup. Batch size and iteration count are implementation choices because the paper does not report them.

The GBM [reconstruction and prior figure](outputs/latent_gbm_figure.png) includes a 95% posterior band from 512 samples. The Lorenz [main figure](outputs/lorenz_context1/latent_lorenz_figure.png) and [extended figure](outputs/lorenz_context1/latent_lorenz_extended_figure.png) show new trajectories, reconstructions, and prior samples. Descriptive posterior-mean MSE on **16 training paths** was **0.00121** for GBM and **0.00581** for Lorenz. These are not held-out scores or numbers claimed by the paper. They support a qualitative reproduction of the figure content, not exact numerical agreement.

## 4. CMU motion forecasting (Section 7.3)

The ODE2VAE-preprocessed CMU subject 35 data contain **16/3/4 training/validation/test sequences**, each with **300 frames × 50 coordinates**. The encoder receives the first three frames; MSE is evaluated on the remaining 297 frames. Training uses Euler steps of 0.02 (one fifth of the 0.1 observation interval), Adam starting at 0.01 with 0.999 decay, at most 400 iterations, KL weight 0.1 with a 200-iteration warmup, and validation-selected checkpoints. The KL choice, zero initialization of the final drift layers, and gradient clipping are documented implementation choices; the authors did not publish the exact selected configuration. Test evaluation uses 50 independent draws from each fixed checkpoint.

| Model | Best validation MSE | Test MSE, 297-frame posterior forecast |
|---|---:|---:|
| Published latent SDE, Table 2 | — | **4.03 ± 0.20** |
| Initial implementation, learned observation variance | 31.27 | **32.691 ± 0.011** |
| Diagnostic: fixed observation std = 1, time inputs | 18.77 | **23.258 ± 0.033** |
| Diagnostic: fixed std = 1, no explicit time inputs | 24.93 | **16.335 ± 0.257** |
| Zero forecast, same local test data | — | **33.675** |

The local `±` values are **95% t half-widths over 50 Monte Carlo forecast draws at a fixed checkpoint**. They do not quantify uncertainty across training runs or data splits, and the paper does not fully specify how it formed its interval. No diagnostic row is an exact reproduction of the published model. The no-time model has a worse *validation* score than the time-input model despite a better *test* score; with only three validation and four test sequences, selecting a model by this test ranking would be inappropriate.

The initial implementation does not reach the paper's Table 2 performance. Its observation variance expands while the forecast mean stays near zero. Fixing the variance improves the fit, but long-horizon errors remain substantial: the time-input model's test MSE rises from **11.38** over frames 3–49 to **35.65** over frames 200–299. Its decoder output then becomes nearly constant. The no-time model retains a periodic forecast but still has a **21.92** MSE over frames 200–299. See [MOCAP_AUDIT.md](MOCAP_AUDIT.md) for the parameter-count discrepancy, controlled tests, and mechanism checks.

[Initial forecast](outputs/mocap_full_400/mocap_prediction_posterior.png) · [time-input diagnostic](outputs/mocap_diagnostics/fixedstd1_full300/mocap_prediction_posterior.png) · [no-time diagnostic](outputs/mocap_diagnostics/fixedstd1_no_time_full300/mocap_prediction_posterior.png) · [framewise error](outputs/mocap_diagnostics/fixedstd1_full300/mocap_error_by_frame_posterior.png).

## Conclusion

The numerical studies reproduce the expected decline in gradient error with solver step size and show the associated computation costs. Independently trained latent SDEs reproduce the synthetic examples' qualitative figures. The motion-capture **protocol** was reconstructed, but its published **MSE was not**. Our evidence identifies variance-driven optimization failure and long-horizon decoder behavior in our implementations; missing author code and conflicting architecture information prevent attribution of the entire numerical gap to any single cause.

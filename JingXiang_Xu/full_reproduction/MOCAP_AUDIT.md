# Why the motion-capture score was not reproduced

This audit concerns Section 7.3 and Table 2 of [Li et al. (2020)](https://proceedings.mlr.press/v108/li20i.html). It compares the [complete supplement, Sections 9.15–9.16 and Figure 10](https://proceedings.mlr.press/v108/li20i/li20i-supp.pdf), the public [ODE2VAE implementation and data](https://github.com/cagatayyildiz/ODE2VAE), and our recorded runs. It separates facts established by the sources from diagnoses specific to our implementation.

## Protocol and unresolved specifications

The public `mocap35.mat` file matches the reported 16/3/4 split and 300-frame, 50-coordinate sequences. We encode the first three frames and forecast the remaining 297. The observation interval is 0.1 in the source data; our Euler step is 0.02, one fifth of that interval. This reproduces the available evaluation geometry.

Three implementation details remain consequential:

1. **Variance parameterization.** The supplement describes encoder log-variance and refers to the ODE2VAE Gaussian observation likelihood. The [ODE2VAE source](https://github.com/cagatayyildiz/ODE2VAE) converts log-variance to standard deviation with `exp(logvar/2)`. Our initial implementation treated both network outputs as log-standard-deviations. The corrected interpretation is used in the diagnostic code.
2. **Network size.** Counting the layers drawn in Figure 10 gives **11,845** trainable parameters. The supplement states **11,605**. Removing the time input from the prior drift, posterior drift, and six scalar diffusion networks removes exactly **8 × 30 = 240** weights. This reconciles the count but contradicts the figure; neither source establishes the authors' actual code.
3. **Training and prediction choices.** The paper lists KL coefficients 1, 0.1, 0.01, and 0.001, with and without a 200-iteration warmup, but does not identify the final choice or release its mocap training/evaluation code. Our choice of 0.1 with warmup, zero final-drift initialization, gradient clipping, and the posterior-drift forecast are explicit local choices. Prior-drift scores are retained in the JSON files as a sensitivity check.

## Controlled diagnosis

The initial 400-iteration checkpoint has training reconstruction MSE **32.28** and posterior test MSE **32.69**, close to the **33.67** zero-forecast test baseline. Its predicted observation standard deviation averages **3.62** and reaches **20.09**. Gaussian negative log-likelihood contains `(x−mean)²/(2 std²) + log(std)`, so an enlarged `std` can reduce the objective while the mean trajectory remains inaccurate. The [likelihood measurements](outputs/mocap_full_400/likelihood_diagnosis.json) and the following controlled comparisons support this mechanism for our model.

All short comparisons use the same split, seed, first-three-frame input, Euler step, and at most 100 iterations. They evaluate **47 future frames only**; their numbers are diagnostic and cannot be compared with the published 297-frame MSE.

| Short-run change | Validation MSE at selected checkpoint | Test MSE |
|---|---:|---:|
| Correct log-variance interpretation; learned observation variance | 30.94 | 32.18 |
| Same model, no explicit time inputs (11,605 parameters) | 30.95 | 32.18 |
| Learned observation std capped at 2 | 10.00 | 7.38 |
| Observation std fixed at 1 | 9.96 | 6.65 |

The checkpoint is selected using the validation curve; the table reports a separate sampled evaluation of that checkpoint, so its validation values need not equal the curve's minima. The first two models select their checkpoint after the first iteration. Thus correcting log-variance and matching the printed parameter count alone do not resolve the short-run optimization failure. Limiting variance does. The [validation curves](outputs/mocap_diagnostics/mocap_ablation_50.png) and [CSV](outputs/mocap_diagnostics/mocap_ablation_50.csv) preserve the comparison.

We then trained the two fixed-variance models for the full **300 frames and 400 iterations** and evaluated **50 test draws**. The time-input model obtained **23.258 ± 0.033** posterior MSE; the no-time model obtained **16.335 ± 0.257**. These changes improve our result but depart from the supplement's referenced learnable Gaussian likelihood, and both remain well above **4.03 ± 0.20**. Their validation ranking is the reverse of their test ranking, so the no-time version cannot be selected as a faithful paper architecture on the basis of test MSE.

The stated parameter counts include the decoder's variance-output weights. Those weights remain registered but receive no gradient when observation std is fixed; the fixed-variance models therefore also differ in effective trainable capacity.

For the time-input model, the posterior test error rises from **11.38** at frames 3–49 to **35.65** at frames 200–299. A fixed-seed trajectory check shows that latent states still change by a mean absolute **0.541 per step** after frame 200, while decoded observations change by only **8.8×10⁻⁷ per step**. All 30 units of the decoder's second Softplus layer are below 0.001 at frames 249 and 299 on the checked paths. The flat long-horizon forecast is therefore consistent with **decoder saturation**, not halted stochastic motion. The no-time version continues to change but has phase and amplitude errors. These are diagnoses of our trained checkpoints, not claims about the authors' model.

The failure is not unique to this attempt: a [TorchSDE discussion](https://github.com/google-research/torchsde/issues/112) reports a separate 20–30 MSE reproduction, and a [later NeurIPS study, Appendix E.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/18abbeef8cfe9203fdf9053c9c4fe191-Paper-Conference.pdf) also reports difficulty matching 4.03. These reports add context but do not identify the missing author settings. The defensible conclusion is that we reproduced the data protocol and investigated the gap, **not** that we reproduced Table 2 numerically.

## TorchSDE adjoint comparison and gradient verification

We tested the stochastic adjoint implementation supplied by TorchSDE on the one-dimensional Itô geometric Brownian motion (GBM) model

\[
dX_t=\mu X_t\,dt+\sigma X_t\,dW_t,\qquad X_0=1,
\]

with \(\mu=0.18\), \(\sigma=0.35\), \(T=1\), and the terminal loss \(L=B^{-1}\sum_{i=1}^B(X_T^{(i)}-1.30)^2\). We used \(B=256\) paths and float64 arithmetic. Both `torchsde.sdeint` and `torchsde.sdeint_adjoint` used an Euler forward solver and the same `BrownianInterval` realization, seeded with 5600. The adjoint solve also used Euler. This controls for Monte Carlo noise when comparing their gradients.

The direct backpropagation gradient is the derivative of the *discrete* Euler loss. To verify it independently, we differentiated each Euler step. If \(F_k=1+\mu\Delta t+\sigma\Delta W_k\), then \(X_{k+1}=X_kF_k\) and

\[
\partial_\mu X_{k+1}=F_k\partial_\mu X_k+X_k\Delta t,\qquad
\partial_\sigma X_{k+1}=F_k\partial_\sigma X_k+X_k\Delta W_k.
\]

We also checked the continuous-time solution \(X_T=X_0\exp\{(\mu-\sigma^2/2)T+\sigma W_T\}\), for which \(\partial_\mu X_T=TX_T\) and \(\partial_\sigma X_T=(W_T-\sigma T)X_T\). These exact-solution derivatives are a convergence reference, while the stepwise Euler derivatives are the correct reference for the discretized solver.

| Euler steps | Backprop \(\partial_\mu L\) | Adjoint \(\partial_\mu L\) | Backprop \(\partial_\sigma L\) | Adjoint \(\partial_\sigma L\) | Maximum adjoint/backprop gap |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.225800 | 0.250241 | 1.572459 | 1.579866 | 0.024441 |
| 32 | 0.254611 | 0.262918 | 1.601356 | 1.614135 | 0.012778 |
| 64 | 0.274294 | 0.278929 | 1.634255 | 1.648450 | 0.014195 |
| 128 | 0.285867 | 0.287838 | 1.661045 | 1.663945 | 0.002900 |
| 256 | 0.293540 | 0.295517 | 1.696544 | 1.694305 | 0.002239 |

The exact GBM gradient for the same Brownian paths was \((\partial_\mu L,\partial_\sigma L)=(0.297170,1.701558)\). The two TorchSDE methods generated the same forward paths to within \(4\times10^{-14}\). Standard backpropagation agreed with the independently computed discrete Euler gradient to within \(3\times10^{-14}\). At 256 steps, central finite differences of the TorchSDE forward loss agreed with backpropagation to within \(10^{-9}\). The adjoint/backpropagation gap fell from 0.024441 at 16 steps to 0.002239 at 256 steps, although it increased locally between 32 and 64 steps.

The adjoint gradient need not equal the exact derivative of a fixed Euler discretization: the backward sensitivity dynamics are solved numerically as a separate SDE. The experiment supports the practical consistency of the adjoint computation under refinement in this GBM example, but does not measure memory use or runtime and cannot establish a general performance advantage. The group's earlier GBM script uses the exact exponential update and different sampled noise, so its raw gradient values are not directly comparable with these Euler results.

**References.** Li, X., Wong, T.-K. L., Chen, R. T. Q., and Duvenaud, D. (2020), “Scalable Gradients for Stochastic Differential Equations,” *AISTATS*, Section 3, https://proceedings.mlr.press/v108/li20i.html. TorchSDE documentation, https://github.com/google-research/torchsde/blob/master/DOCUMENTATION.md.

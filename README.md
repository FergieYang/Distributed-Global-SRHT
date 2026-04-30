# Distributed Global Subsampled Randomized Hadamard Transform (DGSRHT)

**The Merits of Randomized Hadamard Transform in Distributed Regression via Partitioned Machines**

*Yishu Yang, Center for Data Science, New York University*
*Supervised by Prof. Zhixiang Zhang, University of Macau*

## Overview

This repository contains the paper and simulation code for a research project on improving the relative efficiency of distributed Ordinary Least Squares (OLS) regression using the Randomized Hadamard Transform (RHT). In large-scale distributed machine learning, data is partitioned across multiple machines and local OLS estimators are averaged. The key challenge is that naive uniform partitioning can yield poor relative efficiency when data distributions are heterogeneous—particularly under Gaussian Mixture Models (GMM) with biased variance across rows.

## Key Contributions

1. **Identifying the failure mode of uniform sampling**: We show that when data follows a GMM distribution with heterogeneous variance components, uniform partitioning causes drastic drops in relative efficiency because the local Gram matrices `X_i^T X_i` vary widely across machines.

2. **RHT as a variance equalizer**: By applying the Randomized Hadamard Transform (`X_tilde = H_n D X`) before partitioning, the row norms and leverage scores are flattened, making the local Gram matrices nearly identical across all machines. This eliminates the need for complex optimal weight adjustments—simple equal-weight averaging (`w_i = 1/K`) suffices.

3. **Asymptotic optimality (Main Result — Corollary 4.1)**: We rigorously prove that under controlled intermediate growth conditions on `p(n)`:
   - **(C1)** `ln(n) = o(p(n))` — the dimension is superlogarithmic in `n`
   - **(C2)** `p(n) ln(p(n)) = o(n)` — the dimension is sublinear in `n`

   the relative efficiency of the equal-weight distributed OLS estimator after RHT converges to the ideal value of **1** as `n → ∞`. The convergence rate of the trace difference is `O(sqrt(p^3 log(n) / n^3))`.

4. **DGSRHT Algorithm**: A practical distributed algorithm where each client applies a local RHT, the server broadcasts a shared subsampling mask, and a server-side Hadamard mixing step produces the final sketched design matrix—all without sharing raw data.

## Simulation Results

The simulation (`Fergie/simulation.ipynb`) validates the theoretical findings under three heterogeneity scenarios with parameters `n=30,000`, `k=16` clients, `p=200` features, and subsampling proportion `1/s = 0.25`:

### Key Findings

- **DGSRHT consistently achieves near-perfect relative efficiency (~1.0)** across all numbers of working machines `K'` from 1 to 30, in all three heterogeneity scenarios. The empirical results closely match the theoretical predictions.

- **Uniform sampling degrades significantly as `K'` increases**: Under the GMM distribution, the relative efficiency of uniform sampling drops as more machines are used for the distributed OLS step, because the variance heterogeneity in the original data leads to imbalanced local Gram matrices.

- **Robustness across heterogeneity levels**:
  - **Homogeneous mode**: All clients share the same GMM parameters. DGSRHT achieves RE ≈ 1.0; uniform sampling also performs reasonably but still below DGSRHT.
  - **Weight-only heterogeneity**: Per-client mixture weights vary via Beta(8, 2). DGSRHT maintains RE ≈ 1.0 while uniform sampling shows moderate efficiency loss.
  - **Full heterogeneity**: Per-client weights, means, covariances, and scales all vary. DGSRHT still maintains RE ≈ 1.0, demonstrating strong robustness, while uniform sampling suffers more pronounced efficiency degradation.

- **Theory-empirical agreement**: The theoretical relative efficiency curves (computed from the trace formula) closely track the empirical curves (averaged over 50 Monte Carlo repetitions), confirming the validity of the analytical results.

## Repository Structure

```
Fergie/
├── paper_LaTeX.tex              # Main paper (LaTeX source)
├── paper_LaTeX.pdf              # Compiled paper
├── simulation.ipynb             # Simulation notebook (Python/NumPy)
├── algorithm_provement.tex      # Detailed proofs
├── simulation_explanation.tex   # Simulation methodology explanation
├── biblio.bib                   # Bibliography
├── sample.bib                   # Additional references
├── nejsds.cls                   # Journal style file
└── nessart-number.bst           # BibTeX style
```

## How to Run the Simulation

```bash
pip install numpy matplotlib jupyter
cd Fergie
jupyter notebook simulation.ipynb
```

The notebook implements:
- Walsh-Hadamard Transform (FWHT) for local RHT
- GMM data generation with configurable heterogeneity
- The full DGSRHT pipeline (local shuffle → local RHT → shared subsampling → server Hadamard mixing)
- Uniform sampling baseline for comparison
- Relative efficiency computation (both empirical and theoretical)

## References

- Dobriban, E. and Sheng, S. (2022). *Distributed linear regression by averaging.* The Annals of Statistics.
- Tropp, J.A. (2011). *Improved analysis of the subsampled randomized Hadamard transform.* Advances in Adaptive Data Analysis.
- Cherapanamjeri, Y. et al. (2022). *Uniform approximations for randomized Hadamard transforms.* STOC.
- Vershynin, R. (2018). *High-Dimensional Probability.* Cambridge University Press.

## License

This project is for academic research purposes.

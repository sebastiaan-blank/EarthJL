# EarthJL

Julia package for Multivariate Adaptive Regression Splines (MARS) via Stephen
Milborrow's standalone [earth](https://cran.r-project.org/package=earth) C library
(v5.3.6), with term selection and coefficient estimation performed by Julia
[GLMNet](https://github.com/JuliaStats/GLMNet.jl) (lasso with cross-validation).

Supports continuous and binary outcomes, is thread-safe, and is designed for use as
a learner inside SuperLearner ensembles.

## Installation

EarthJL is not yet registered. Install from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/sebastiaan-blank/EarthJL")
```

Before using, build the C library (requires gcc and OpenBLAS):

```bash
cd ~/.julia/packages/EarthJL/<hash>/deps/earth
make
```

Or if working from the source directory:

```bash
cd deps/earth && make
```

## Quick start

```julia
using EarthJL

# Fit on continuous outcome
fit = EarthJL.fit(X, y)               # X: Matrix{Float64}, y: Vector{Float64}

# Fit on binary outcome
fit = EarthJL.fit(X, y, family = :binomial)

# Predict
yhat = EarthJL.predict(fit, X_new)
```

### Pruning methods

| `pmethod` | Forward pass | Term selection |
|-----------|-------------|----------------|
| `:cv`     | Earth (no GCV prune) | GLMNet lasso CV |
| `:gcv`    | Earth GCV prune | GLMNet lasso CV on pruned basis |

`:cv` is recommended for binary outcomes and density ratio estimation; GLMNet's
lasso regularisation prevents extreme predictions that unregularised pruning can
produce.

### Key parameters

```julia
EarthLearner(
    degree   = 2,        # maximum interaction degree
    nk       = 21,       # maximum basis functions (nk + 1 including intercept)
    penalty  = 3.0,      # GCV complexity penalty per knot (earth default)
    thresh   = 0.001,    # forward pass R² improvement threshold
    fast_k   = 20,       # Fast MARS candidate knot limit (0 = disabled)
    pmethod  = :cv,      # :cv or :gcv
    nfolds   = 10,       # GLMNet cross-validation folds
)
```

### Thread safety

All 29 mutable globals in `earth.c` are patched to `__thread` TLS storage, making
concurrent calls from Julia multi-threaded code safe. See `deps/earth/earth.c` for
patch details.

## References

- Milborrow S (2024). *earth: Multivariate Adaptive Regression Splines*.
  R package version 5.3.6. CRAN. https://cran.r-project.org/package=earth.
  Derived from mda:mars by Hastie T and Tibshirani R.

- Friedman JH (1991). Multivariate Adaptive Regression Splines (with discussion).
  *Annals of Statistics* 19(1):1–141.

- Friedman JH (1993). Fast MARS. Department of Statistics, Stanford University,
  Technical Report 110.

- Hastie T, Tibshirani R (1994). S library mda version 0.3.2, file dmarss.r (Ratfor).

- Miller A (2002). *Subset Selection in Regression*, 2nd ed. Chapman & Hall/CRC.
  (Used for QR-based coefficient estimation in earth.)

## License

GPL-2 or later — the same license as the upstream earth C library. See `LICENSE`.

## Acknowledgements

The core C implementation is due to Stephen Milborrow (earth R package). The LINPACK
routines bundled in `deps/earth/blas/` and `deps/earth/R/` are derived from
LINPACK (Dongarra et al.) and R's modified QR decomposition (`dqrdc2.c`).

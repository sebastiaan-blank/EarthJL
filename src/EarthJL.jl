module EarthJL

import GLMNet
using Statistics: mean

const _libearth = joinpath(@__DIR__, "..", "deps", "earth", "libearth.so")

# ==================================================================
# EarthJL — MARS via standalone earth C library + Julia GLMNet
#
# Wraps Stephen Milborrow's earth 5.3.6 C library (GPL-2+).
# Forward pass in C; term selection and coefficient fitting via
# Julia GLMNet (lasso CV), which is binomial-aware.
#
# pmethod = :cv   — forward pass only (Prune=false), GLMNet CV
#                   selects terms. Recommended for binary outcomes.
# pmethod = :gcv  — Earth GCV backward pass selects terms, then
#                   GLMNet CV fits on that basis.
#
# Thread-safe: all 29 mutable C globals patched to __thread TLS.
# ABI: bool is _Bool (1 byte) from <stdbool.h> → UInt8 in Julia.
# ==================================================================

struct EarthLearner
    degree    :: Int
    nk        :: Int        # nMaxTerms = nk + 1 (intercept)
    penalty   :: Float64
    thresh    :: Float64
    fast_k    :: Int
    pmethod   :: Symbol     # :cv or :gcv
    nfolds    :: Int
    name      :: String
end

struct _MeanFallback
    mean_y :: Float64
end

struct _GLMNetResult
    path       :: GLMNet.GLMNetPath
    lambda_idx :: Int
end

struct EarthFit
    active_set  :: Vector{Bool}
    dirs        :: Matrix{Int32}
    cuts        :: Matrix{Float64}
    n_max_terms :: Int
    n_preds     :: Int
    glmnet_fit  :: Any      # _GLMNetResult or _MeanFallback
    family      :: Symbol
end

# ------------------------------------------------------------------
# ccall to Earth()
# ------------------------------------------------------------------

function _earth_call(x::Matrix{Float64}, y::Vector{Float64},
                     max_degree::Int, max_terms::Int,
                     penalty::Float64, thresh::Float64,
                     fast_k::Int, prune::Bool)
    nCases, nPreds = size(x)

    best_gcv    = Ref{Float64}(0.0)
    n_terms_out = Ref{Int32}(0)
    term_cond   = Ref{Int32}(0)                          # out: termination reason
    # earth.c 5.3.6 uses <stdbool.h> → bool is _Bool (1 byte): UInt8 arrays.
    best_set    = zeros(UInt8,   max_terms)
    bx          = zeros(Float64, nCases * max_terms)
    dirs        = zeros(Int32,   max_terms * nPreds)
    cuts        = zeros(Float64, max_terms * nPreds)
    residuals   = zeros(Float64, nCases)
    betas       = zeros(Float64, max_terms)
    lin_preds   = zeros(Int32,   nPreds)

    ccall((:Earth, _libearth), Cvoid,
        (Ptr{Float64}, Ptr{Int32}, Ptr{Int32},            # pBestGcv, pnTerms, piTermCond
         Ptr{UInt8},   Ptr{Float64},                      # BestSet, bx
         Ptr{Int32},   Ptr{Float64},                      # Dirs, Cuts
         Ptr{Float64}, Ptr{Float64},                      # Residuals, Betas
         Ptr{Float64}, Ptr{Float64}, Ptr{Cvoid},          # x, y, WeightsArg
         Csize_t, Int32, Int32,                           # nCases (size_t), nResp, nPreds
         Int32, Int32,                                    # nMaxDegree, nMaxTerms
         Float64, Float64,                                # Penalty, Thresh
         Int32, Int32,                                    # nMinSpan, nEndSpan
         UInt8,                                           # Prune
         Int32, Float64, Float64,                         # nFastK, FastBeta, NewVarPenalty
         Ptr{Int32},                                      # LinPreds
         Float64, UInt8,                                  # AdjustEndSpan, AutoLinPreds
         UInt8, Int32, Float64, Ptr{Cvoid}),              # UseBetaCache, nThreads, Trace, sPredNames
        best_gcv, n_terms_out, term_cond,
        best_set, bx, dirs, cuts, residuals, betas,
        x, y, C_NULL,
        Csize_t(nCases), Int32(1), Int32(nPreds),
        Int32(max_degree), Int32(max_terms),
        penalty, thresh,
        Int32(0), Int32(0),                               # nMinSpan, nEndSpan (0 = auto)
        UInt8(prune),
        Int32(fast_k), Float64(0.0), Float64(0.0),
        lin_preds,
        Float64(2.0), UInt8(1),                           # AdjustEndSpan, AutoLinPreds (R defaults)
        UInt8(1), Int32(1), Float64(0.0), C_NULL)         # UseBetaCache=1, nThreads=1 (reserved), Trace=0

    n_terms  = Int(n_terms_out[])
    bx_mat   = reshape(bx,   nCases,    max_terms)
    dirs_mat = reshape(dirs, max_terms, nPreds)
    cuts_mat = reshape(cuts, max_terms, nPreds)

    n_terms, best_set .!= 0, bx_mat, dirs_mat, cuts_mat
end

# ------------------------------------------------------------------
# Extract active basis columns; called inside a let block so bx_mat
# is freed before GLMNet allocates.
# ------------------------------------------------------------------

function _earth_extract_basis(bx_mat, best_set, dirs_mat, cuts_mat,
                               max_terms, use_gcv)
    active_set = use_gcv ?
        best_set :
        Bool[any(!=(0.0), view(bx_mat, :, i)) for i in 1:max_terms]
    n_used = sum(active_set)
    nCases = size(bx_mat, 1)
    B = zeros(Float64, nCases, n_used)
    col = 1
    for iTerm in 1:max_terms
        active_set[iTerm] || continue
        B[:, col] = bx_mat[:, iTerm]
        col += 1
    end
    active_set, B, copy(dirs_mat), copy(cuts_mat)
end

# ------------------------------------------------------------------
# Evaluate MARS basis for new X from stored knot info
# ------------------------------------------------------------------

function _earth_basis(X::Matrix{Float64}, active_set::Vector{Bool},
                      dirs::Matrix{Int32}, cuts::Matrix{Float64},
                      n_max_terms::Int)
    nCases = size(X, 1)
    n_used = sum(active_set)
    B      = ones(Float64, nCases, n_used)
    col    = 1
    for iTerm in 1:n_max_terms
        active_set[iTerm] || continue
        if iTerm == 1
            col += 1; continue
        end
        b = ones(Float64, nCases)
        for iPred in axes(X, 2)
            d = dirs[iTerm, iPred]
            d == 0 && continue
            c = cuts[iTerm, iPred]
            if d == 2
                b .*= view(X, :, iPred)
            elseif d == 1
                b .*= max.(0.0, view(X, :, iPred) .- c)
            elseif d == -1
                b .*= max.(0.0, c .- view(X, :, iPred))
            end
        end
        B[:, col] = b
        col += 1
    end
    B
end

# ------------------------------------------------------------------
# fit / predict
# ------------------------------------------------------------------

"""
    fit(l::EarthLearner, Y, X, family, rng) → EarthFit

Fit a MARS model. `family` is `:binomial` or `:gaussian`.
"""
function fit(l::EarthLearner, Y, X, family::Symbol, rng)
    Xmat = Matrix{Float64}(X)
    Yf   = Float64.(Y)
    nCases, nPreds = size(Xmat)
    max_terms = l.nk + 1
    use_gcv   = l.pmethod == :gcv

    # Phase 1: C forward pass — large bx_mat scoped in let block
    active_set, B, dirs_mat, cuts_mat = let
        _, best_set, bx_mat, dm, cm =
            _earth_call(Xmat, Yf, l.degree, max_terms,
                        l.penalty, l.thresh, l.fast_k, use_gcv)
        _earth_extract_basis(bx_mat, best_set, dm, cm, max_terms, use_gcv)
    end

    n_used = sum(active_set)
    if n_used <= 1
        return EarthFit(active_set, dirs_mat, cuts_mat, max_terms, nPreds,
                        _MeanFallback(mean(Yf)), family)
    end

    # Phase 2: GLMNet CV on basis
    GC.gc(false)
    cv = if family == :binomial
        GLMNet.glmnetcv(B, hcat(1.0 .- Yf, Yf), GLMNet.Binomial();
                        alpha = 1.0, nfolds = l.nfolds)
    else
        GLMNet.glmnetcv(B, Yf; alpha = 1.0, nfolds = l.nfolds)
    end
    lam_idx = argmin(cv.meanloss)

    EarthFit(active_set, dirs_mat, cuts_mat, max_terms, nPreds,
             _GLMNetResult(cv.path, lam_idx), family)
end

"""
    predict(f::EarthFit, X, family) → Vector{Float64}
"""
function predict(f::EarthFit, X, family::Symbol)
    Xmat = Matrix{Float64}(X)
    if f.glmnet_fit isa _MeanFallback
        p = fill(f.glmnet_fit.mean_y, size(Xmat, 1))
        return family == :binomial ? clamp.(p, 1e-5, 1.0 - 1e-5) : p
    end
    B     = _earth_basis(Xmat, f.active_set, f.dirs, f.cuts, f.n_max_terms)
    col   = min(f.glmnet_fit.lambda_idx, length(f.glmnet_fit.path.a0))
    preds = vec(GLMNet.predict(f.glmnet_fit.path, B, col; outtype = :prob))
    family == :binomial ? clamp.(preds, 1e-5, 1.0 - 1e-5) : preds
end

# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------

"""
    sl_earth(; degree=4, nk=600, penalty=2.0, pmethod=:cv, nfolds=5, name=nothing)

MARS learner via standalone earth C library + Julia GLMNet. Thread-safe.
`pmethod=:cv`  — GLMNet CV selects basis terms (recommended for binary outcomes).
`pmethod=:gcv` — Earth GCV backward pass, then GLMNet CV on pruned basis.
"""
function sl_earth(;
    degree  = 4,
    nk      = 600,
    penalty = 2.0,
    thresh  = 0.001,
    fast_k  = 20,
    pmethod = :cv,
    nfolds  = 5,
    name    = nothing
)
    nm = isnothing(name) ? "earth_d$(degree)_nk$(nk)_$(pmethod)" : name
    EarthLearner(degree, nk, penalty, thresh, fast_k, pmethod, nfolds, nm)
end

const sl_earth_d4      = sl_earth(degree=4, nk=600, name="earth_d4_nk600")
const sl_earth_d4_fast = sl_earth(degree=4, nk=300, name="earth_d4_nk300")
const sl_earth_d3      = sl_earth(degree=3, nk=600, name="earth_d3_nk600")
const sl_earth_d2      = sl_earth(degree=2, nk=600, name="earth_d2_nk600")

export EarthLearner, EarthFit,
       sl_earth, sl_earth_d4, sl_earth_d4_fast, sl_earth_d3, sl_earth_d2,
       fit, predict

end # module

#
# Circular mean of phase difference
#

immutable MeanPhaseDiff <: ComplexPairwiseStatistic
    normalized::Bool
end
MeanPhaseDiff() = MeanPhaseDiff(false)

immutable MeanPhaseDiffWork{T<:Real}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    MeanPhaseDiffWork() = new()
    MeanPhaseDiffWork(x) = new(x)
    MeanPhaseDiffWork(x, y) = new(x, y)
end

# Single input matrix
allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() : MeanPhaseDiffWork{T}(Array(Complex{T}, size(X)))
_finish(::MeanPhaseDiff, x, n) = x/n
function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                               work::MeanPhaseDiffWork{T}, X::AbstractVecOrMat{Complex{T}})
    scale!(Ac_mul_A!(out, normalized(t, work, X)), 1/ntrials(X))
end

# Two input matrices
allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() :
        MeanPhaseDiffWork{T}(Array(Complex{T}, size(X)), Array(Complex{T}, size(Y)))
_finish_xy!(::MeanPhaseDiff, out, work, n) = scale!(out, work, 1/n)
function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                               work::MeanPhaseDiffWork{T}, X::AbstractVecOrMat{Complex{T}})
    X, Y = normalized(t, work, X, Y)
    scale!(Ac_mul_B!(out, X, Y), 1/ntrials(X))
end

#
# Phase locking value, pairwise phase consistency
#
# For PLV, see Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela,
# F. J. (1999). Measuring phase synchrony in brain signals. Human Brain
# Mapping, 8(4), 194–208.
# doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C
#
# For PPC, see Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P.,
# & Pennartz, C. M. A. (2010). The pairwise phase consistency: A
# bias-free measure of rhythmic neuronal synchronization. NeuroImage,
# 51(1), 112–122. doi:10.1016/j.neuroimage.2010.01.073

immutable PLV <: RealPairwiseStatistic
    normalized::Bool
end
PLV() = PLV(false)

immutable PPC <: RealPairwiseStatistic
    normalized::Bool
end
PPC() = PPC(false)

immutable PLVWork{T<:Real}
    meanphasediff::Matrix{Complex{T}}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    PLVWork(w1) = new(w1)
    PLVWork(w1, w2) = new(w1, w2)
    PLVWork(w1, w2, w3) = new(w1, w2, w3)
end

# Computes the result value based on the element of the matrix product
# (x) and the number of samples (n)
_finish(::PLV, x, n) = abs(x)/n
_finish(::PPC, x, n) = abs2(x)/(n*(n-1)) - 1/(n-1)

# Single input matrix
allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? PLVWork{T}(Array(Complex{T}, nchannels(X), nchannels(X))) :
        PLVWork{T}(Array(Complex{T}, nchannels(X), nchannels(X)),
                   Array(Complex{T}, size(X, 1), size(X, 2)))
function _finish!(t::PairwiseStatistic, out, work, n)
    for j = 1:size(out, 1), i = 1:j
        @inbounds out[i, j] = _finish(t, work[i, j], n)
    end
    out
end
function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T},
                               X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    _finish!(t, out, Ac_mul_A!(work.meanphasediff, normalized(t, work, X)), ntrials(X))
end

# Two input matrices
allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? PLVWork{T}(Array(Complex{T}, nchannels(X), nchannels(Y))) :
        PLVWork{T}(Array(Complex{T}, nchannels(X), nchannels(Y)),
                   Array(Complex{T}, size(X, 1), size(X, 2)),
                   Array(Complex{T}, size(Y, 1), size(Y, 2)))
function _finish_xy!(t::PairwiseStatistic, out, work, n)
    for i = 1:length(out)
        @inbounds out[i] = _finish(t, work[i], n)
    end
    out
end
function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    X, Y = normalized(t, work, X, Y)
    _finish_xy!(t, out, Ac_mul_B!(work.meanphasediff, X, Y), ntrials(X))
end

#
# Jackknifing for MeanPhaseDiff/PLV/PPC
#

# Single input matrix
allocwork{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                   X::AbstractVecOrMat{Complex{T}}) = allocwork(PLV(), X)
function computestat!{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                               out::JackknifeOutput,
                               work::PLVWork{T}, X::AbstractVecOrMat{Complex{T}})
    X = normalized(t.transform, work, X)
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc = work.meanphasediff

    chkinput(trueval, X)

    Ac_mul_A!(XXc, X)

    n = ntrials(X)
    _finish!(t.transform, trueval, XXc, n)

    invnm1 = inv(n-1)
    invnm2 = inv(n-2)
    invppc = inv((n-1)*(n-2))
    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1
            x = XXc[j, k]
            for i = 1:size(X, 1)
                v = x - conj(X[i, j])*X[i, k]
                if isa(t, Jackknife{MeanPhaseDiff})
                    surrogates[i, j, k] = v*invnm1
                elseif isa(t, Jackknife{PLV})
                    surrogates[i, j, k] = abs(v)*invnm1
                elseif isa(t, Jackknife{PPC})
                    surrogates[i, j, k] = abs2(v)*invppc - invnm2
                end
            end
        end
        for i = 1:size(X, 1)
            surrogates[i, k, k] = 1
        end
    end

    out
end

# Two input matrices
allocwork{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                   X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    allocwork(PLV(), X, Y)
function computestat!{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                               out::JackknifeOutput,
                               work::PLVWork{T}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    X, Y = normalized(t.transform, work, X, Y)
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XYc = work.meanphasediff

    chkinput(trueval, X, Y)

    Ac_mul_B!(XYc, X, Y)

    n = ntrials(X)
    _finish_xy!(t.transform, trueval, XYc, n)

    invnm1 = inv(n-1)
    invnm2 = inv(n-2)
    invppc = inv((n-1)*(n-2))
    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2)
        x = XYc[j, k]
        for i = 1:size(X, 1)
            v = x - conj(X[i, j])*Y[i, k]
            if isa(t, Jackknife{MeanPhaseDiff})
                surrogates[i, j, k] = v*invnm1
            elseif isa(t, Jackknife{PLV})
                surrogates[i, j, k] = abs(v)*invnm1
            elseif isa(t, Jackknife{PPC})
                surrogates[i, j, k] = abs2(v)*invppc - invnm2
            end
        end
    end

    out
end

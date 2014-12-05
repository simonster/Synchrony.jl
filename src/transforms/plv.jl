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

allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractMatrix{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() : MeanPhaseDiffWork{T}(Array(Complex{T}, size(X)))
function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                               work::MeanPhaseDiffWork{T}, X::AbstractMatrix{Complex{T}})
    conjscale!(A_mul_Ac!(out, normalized(t, work, X)), 1/size(X, 2))
end

allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() :
        MeanPhaseDiffWork{T}(Array(Complex{T}, size(X)), Array(Complex{T}, size(Y)))
function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                               work::MeanPhaseDiffWork{T}, X::AbstractMatrix{Complex{T}})
    X, Y = normalized(t, work, X, Y)
    conjscale!(A_mul_Bc!(out, X, Y), 1/size(X, 2))
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

allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractMatrix{Complex{T}}) =
    t.normalized ? PLVWork{T}(Array(Complex{T}, size(X, 1), size(X, 1))) :
        PLVWork{T}(Array(Complex{T}, size(X, 1), size(X, 1)),
                   Array(Complex{T}, size(X)))
function _finish!(::PLV, out, work, n)
    for j = 1:size(out, 1)
        for i = 1:j-1
            @inbounds out[i, j] = abs(work[i, j])/n
        end
        @inbounds out[j, j] = one(eltype(out))
    end
    out
end
function _finish!(::PPC, out, work, n)
    for j = 1:size(out, 1)
        for i = 1:j-1
            @inbounds out[i, j] = abs2(work[i, j])/(n*(n-1)) - 1/(n-1)
        end
        @inbounds out[j, j] = one(eltype(out))
    end
    out
end
function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T},
                               X::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    fill!(out, zero(T))
    _finish!(t, out, A_mul_Ac!(work.meanphasediff, normalized(t, work, X)), size(X, 2))
end

allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    t.normalized ? PLVWork{T}(Array(Complex{T}, size(X, 1), size(Y, 1))) :
        PLVWork{T}(Array(Complex{T}, size(X, 1), size(Y, 1)),
                   Array(Complex{T}, size(X)), Array(Complex{T}, size(Y)))
function _finish_xy!(::PLV, out, work, n)
    for i = 1:length(out)
        @inbounds out[i] = abs(work[i])/n
    end
    out
end
function _finish_xy!(::PPC, out, work, n)
    for i = 1:length(out)
        @inbounds out[i] = abs2(work[i])/(n*(n-1)) - 1/(n-1)
    end
    out
end
function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T},
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    size(out, 1) == size(X, 1) && size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    X, Y = normalized(t, work, X, Y)
    _finish_xy!(t, out, A_mul_Bc!(work.meanphasediff, X, Y), size(X, 2))
end


# Jackknifing for MeanPhaseDiff/PLV/PPC
allocwork{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                   X::AbstractMatrix{Complex{T}}) = allocwork(PLV(), X)
function computestat!{T<:Real}(t::Union(Jackknife{MeanPhaseDiff}, Jackknife{PLV}, Jackknife{PPC}),
                               out::JackknifeOutput,
                               work::PLVWork{T}, X::AbstractMatrix{Complex{T}})
    size(out.trueval, 1) == size(out.trueval, 2) == size(X, 1) || error(DimensionMismatch("out"))
    !t.transform.normalized && (X = unitnormalize!(work.normalizedX, X))

    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc = work.meanphasediff

    A_mul_Bc!(XXc, X, X)

    n = size(X, 2)
    fill!(trueval, zero(T))
    if isa(t, Jackknife{MeanPhaseDiff})
        conjscale!(trueval, XXc, 1/n)
    else
        _finish!(t.transform, trueval, XXc, n)
    end

    invnm1 = inv(n-1)
    invnm2 = inv(n-2)
    invppc = inv((n-1)*(n-2))
    @inbounds for k = 1:size(X, 2)
        surrogates[1, 1, k] = one(T)
        for j = 2:size(X, 1)
            z = conj(X[j, k])
            for i = 1:j-1
                v = XXc[i, j] - X[i, j]*z
                if isa(t, Jackknife{MeanPhaseDiff})
                    surrogates[i, j, k] = conj(v)*invnm1
                elseif isa(t, Jackknife{PLV})
                    surrogates[i, j, k] = abs(v)*invnm1
                elseif isa(t, Jackknife{PPC})
                    surrogates[i, j, k] = abs2(v)*invppc - invnm2
                end
            end
            surrogates[j, j, k] = one(T)
            for i = j+1:size(X, 1)
                surrogates[i, j, k] = zero(T)
            end
        end
    end

    out
end

#
# Phase lag index
#
# See Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).
# Phase lag index: Assessment of functional connectivity from multi
# channel EEG and MEG with diminished bias from common sources.
# Human Brain Mapping, 28(11), 1178–1193. doi:10.1002/hbm.20346

immutable PLI <: RealPairwiseStatistic; end
allocwork{T<:Complex}(::PLI, X::AbstractMatrix{T}, Y::AbstractMatrix{T}=X) = nothing
allocoutput{T<:Real}(::PLI, X::AbstractMatrix{Complex{T}}) =
    Array(T, size(X, 1), size(X, 1))

function _sum_pli!{T}(out::AbstractMatrix{T}, X)
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    fill!(out, zero(T))
    for k = 1:size(X, 2), j = 1:size(X, 1), i = 1:j-1
        @inbounds out[i, j] += sign(imag(conj(X[i, k])*X[j, k]))
    end
    out
end
function computestat!{T<:Real}(::PLI, out::AbstractMatrix{T}, ::Nothing,
                               X::AbstractMatrix{Complex{T}})
    _sum_pli!(out, X)
    for j = 1:size(X, 1), i = 1:j-1
        @inbounds out[i, j] = abs(out[i, j])/size(X, 2)
    end
    out
end

function _sum_pli!{T}(out::AbstractMatrix{T}, X, Y)
    size(out, 1) == size(out, 2) == size(X, 1) == size(Y, 1) || error(DimensionMismatch("out"))
    size(X, 2) == size(Y, 2) || error(DimensionMismatch("X and Y"))
    fill!(out, zero(T))
    for k = 1:size(X, 2), j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds out[i, j] += sign(imag(conj(X[i, k])*Y[j, k]))
    end
    out
end
function computestat!{T<:Real}(::PLI, out::AbstractMatrix{T}, ::Nothing,
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    _sum_pli!(out, X, Y)
    for j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds out[i, j] = abs(out[i, j])/size(X, 2)
    end
    out
end

#
# Unbiased squared phase lag index
#
# See Vinck, M., Oostenveld, R., van
# Wingerden, M., Battaglia, F., & Pennartz, C. M. A. (2011). An
# improved index of phase-synchronization for electrophysiological data
# in the presence of volume-conduction, noise and sample-size bias.
# NeuroImage, 55(4), 1548–1565. doi:10.1016/j.neuroimage.2011.01.055

immutable PLI2Unbiased <: RealPairwiseStatistic; end
allocwork{T<:Complex}(::PLI2Unbiased, X::AbstractMatrix{T}, Y::AbstractMatrix{T}=X) = nothing
function computestat!{T<:Real}(::PLI2Unbiased, out::AbstractMatrix{T}, ::Nothing,
                               X::AbstractMatrix{Complex{T}})
    _sum_pli!(out, X)
    _finish!(PPC(), out, out, size(X, 2))
end
function computestat!{T<:Real}(::PLI2Unbiased, out::AbstractMatrix{T}, ::Nothing,
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    _sum_pli!(out, X, Y)
    _finish_xy!(PPC(), out, out, size(X, 2))
end

#
# Weighted phase lag index
#
# See Vinck et al. (2011) as above.

immutable WPLI <: RealPairwiseStatistic; end
allocwork{T<:Real}(::WPLI, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}=X) =
    Array(T, size(X, 1), size(Y, 1))
function computestat!{T<:Real}(::WPLI, out::AbstractMatrix{T}, work::Matrix{T},
                               X::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    size(work, 1) == size(work, 2) == size(X, 1) || error(DimensionMismatch("work"))

    fill!(out, zero(T))
    fill!(work, zero(T))
    for k = 1:size(X, 2), j = 1:size(X, 1), i = 1:j-1
        @inbounds z = imag(conj(X[i, k])*X[j, k])
        @inbounds out[i, j] += z
        @inbounds work[i, j] += abs(z)
    end
    for j = 1:size(X, 1), i = 1:j-1
        @inbounds out[i, j] = abs(out[i, j])/work[i, j]
    end
    out
end

function computestat!{T<:Real}(::WPLI, out::AbstractMatrix{T}, work::Matrix{T},
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    (size(out, 1) == size(X, 1) && size(out, 2) == size(Y, 1)) ||
        error(DimensionMismatch("out"))
    (size(work, 1) == size(X, 1) && size(work, 2) == size(Y, 1)) ||
        error(DimensionMismatch("work"))
    size(X, 2) == size(Y, 2) || error(DimensionMismatch("X and Y"))

    fill!(out, zero(T))
    fill!(work, zero(T))
    for k = 1:size(X, 2), j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds z = imag(conj(X[i, k])*Y[j, k])
        @inbounds out[i, j] += z
        @inbounds work[i, j] += abs(z)
    end
    for j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds out[i, j] = abs(out[i, j])/work[i, j]
    end
    out
end

#
# Debiased (i.e. still somewhat biased) WPLI^2
#
# See Vinck et al. (2011) as above.

immutable WPLI2Debiased <: RealPairwiseStatistic; end
allocwork{T<:Real}(::WPLI2Debiased, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}=X) =
    Array(T, 2, size(X, 1), size(Y, 1))
function computestat!{T<:Real}(::WPLI2Debiased, out::AbstractMatrix{T}, work::Array{T,3},
                               X::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    size(work, 1) == 2 || error(ArgumentError("first dimension of work must be 2"))
    size(work, 2) == size(work, 3) == size(X, 1) || error(DimensionMismatch("work"))
    fill!(out, zero(T))
    fill!(work, zero(T))
    for k = 1:size(X, 2), j = 1:size(X, 1), i = 1:j-1
        @inbounds z = imag(conj(X[i, k])*X[j, k])
        @inbounds out[i, j] += z
        @inbounds work[1, i, j] += abs(z)
        @inbounds work[2, i, j] += abs2(z)
    end
    for j = 1:size(X, 1), i = 1:j-1 
        @inbounds out[i, j] = (abs2(out[i, j]) - work[2, i, j])/(abs2(work[1, i, j]) - work[2, i, j])
    end
    out
end

function computestat!{T<:Real}(::WPLI2Debiased, out::AbstractMatrix{T}, work::Array{T,3},
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    (size(out, 1) == size(X, 1) && size(out, 2) == size(Y, 1)) ||
        error(DimensionMismatch("out"))
    size(work, 1) == 2 || error(ArgumentError("first dimension of work must be 2"))
    (size(work, 2) == size(X, 1) && size(work, 3) == size(Y, 1)) ||
        error(DimensionMismatch("work"))
    size(X, 2) == size(Y, 2) || error(DimensionMismatch("X and Y"))

    fill!(out, zero(T))
    fill!(work, zero(T))
    for k = 1:size(X, 2), j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds z = imag(conj(X[i, k])*Y[j, k])
        @inbounds out[i, j] += z
        @inbounds work[1, i, j] += abs(z)
        @inbounds work[2, i, j] += abs2(z)
    end
    for j = 1:size(Y, 1), i = 1:size(X, 1)
        @inbounds out[i, j] = (abs2(out[i, j]) - work[2, i, j])/(abs2(work[1, i, j]) - work[2, i, j])
    end
    out
end
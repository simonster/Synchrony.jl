#
# Phase lag index
#
# See Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).
# Phase lag index: Assessment of functional connectivity from multi
# channel EEG and MEG with diminished bias from common sources.
# Human Brain Mapping, 28(11), 1178–1193. doi:10.1002/hbm.20346
immutable PLI <: PairwiseStatistic; end

finish(::Type{PLI}, x::Complex, n::Int) = abs(x)/n

#
# Unbiased squared phase lag index, weighted phase
#
# See Vinck, M., Oostenveld, R., van Wingerden, M., Battaglia, F., &
# Pennartz, C. M. A. (2011). An improved index of phase-synchronization
# for electrophysiological data in the presence of volume-conduction,
# noise and sample-size bias. NeuroImage, 55(4), 1548–1565.
# doi:10.1016/j.neuroimage.2011.01.055
immutable PLI2Unbiased <: PairwiseStatistic; end

accumulator{T<:Real}(::Union{Type{PLI}, Type{PLI2Unbiased}}, ::Type{T}) = zero(Complex{T})
accumulate{T<:Real}(::Union{Type{PLI}, Type{PLI2Unbiased}}, x::Complex{T},
                    v1::Complex{T}, v2::Complex{T}) = (x + sign(imag(conj(v1)*v2)))
accumulate{T<:Real}(::Union{Type{PLI}, Type{PLI2Unbiased}}, x::Complex{T},
                    v1::Complex{T}, v2::Complex{T}, weight::Real) = (x + sign(imag(conj(v1)*v2))*weight)
finish(::Type{PLI2Unbiased}, x::Complex, n::Int) = abs2(x)/(n*(n-1)) - 1/(n-1)

#
# Weighted phase lag index
#
# See Vinck et al. (2011) as above.
immutable WPLI <: PairwiseStatistic; end
immutable WPLIAccumulator{T}
    si::Complex{T}   # Sum of imag(conj(v1)*(v2))
    sa::T            # Sum of abs(imag(conj(v1)*(v2)))
end

accumulator{T<:Real}(::Type{WPLI}, ::Type{T}) =
    WPLIAccumulator{T}(zero(Complex{T}), zero(T))
function accumulate{T<:Real}(::Type{WPLI}, x::WPLIAccumulator{T},
                             v1::Complex{T}, v2::Complex{T})
    z = imag(conj(v1)*v2)
    WPLIAccumulator(x.si + z, x.sa + abs(z))
end
function accumulate{T<:Real}(::Type{WPLI}, x::WPLIAccumulator{T},
                             v1::Complex{T}, v2::Complex{T}, weight::Real)
    z = imag(conj(v1)*v2)
    WPLIAccumulator(x.si + z*weight, x.sa + abs(z*weight))
end
finish(::Type{WPLI}, x::WPLIAccumulator, n::Int) = abs(x.si)/x.sa

#
# Debiased (i.e. still somewhat biased) WPLI^2
#
# See Vinck et al. (2011) as above.
immutable WPLI2Debiased <: PairwiseStatistic; end
immutable WPLI2DebiasedAccumulator{T}
    si::Complex{T}   # Sum of imag(conj(v1)*(v2))
    sa::T            # Sum of abs(imag(conj(v1)*(v2)))
    sa2::T           # Sum of abs2(imag(conj(v1)*(v2)))
end

accumulator{T<:Real}(::Type{WPLI2Debiased}, ::Type{T}) =
    WPLI2DebiasedAccumulator{T}(zero(Complex{T}), zero(T), zero(T))
function accumulate{T<:Real}(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator{T},
                             v1::Complex{T}, v2::Complex{T})
    z = imag(conj(v1)*v2)
    WPLI2DebiasedAccumulator(x.si + z, x.sa + abs(z), x.sa2 + abs2(z))
end
function accumulate{T<:Real}(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator{T},
                             v1::Complex{T}, v2::Complex{T}, weight::Real)
    z = imag(conj(v1)*v2)
    WPLI2DebiasedAccumulator(x.si + z*weight, x.sa + abs(z*weight), x.sa2 + abs2(z)*weight)
end
finish(::Type{WPLI2Debiased}, x::WPLI2DebiasedAccumulator,
       n::Int) = (abs2(x.si) - x.sa2)/(abs2(x.sa) - x.sa2)

#
# Functions applicable to all phase lag-style metrics
#
typealias PLStat Union{PLI, PLI2Unbiased, WPLI, WPLI2Debiased}
Base.eltype{T<:Real}(::PLStat, X::AbstractArray{Complex{T}}) = T
diagval{T<:PLStat}(::Type{T}) = 0

allocwork{T<:Complex}(::PLStat, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}=X) =
    nothing

# Single input matrix
function computestat!{S<:PLStat, T<:Real}(t::S, out::AbstractMatrix{T}, work::Void,
                                          X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    for k = 1:size(X, 2), j = 1:k
        v = accumulator(S, T)
        @simd for i = 1:size(X, 1)
            @inbounds v = accumulate(S, v, X[i, j], X[i, k])
        end
        out[j, k] = finish(S, v, size(X, 1))
    end
    out
end

# Two input matrices
function computestat!{S<:PLStat, T<:Real}(t::S, out::AbstractMatrix{T}, work::Void,
                                          X::AbstractVecOrMat{Complex{T}},
                                          Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    for k = 1:size(Y, 2), j = 1:size(X, 2)
        v = accumulator(S, T)
        @simd for i = 1:size(X, 1)
            @inbounds v = accumulate(S, v, X[i, j], Y[i, k])
        end
        out[j, k] = finish(S, v, size(X, 1))
    end
    out
end

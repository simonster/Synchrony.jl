#
# Jammalamadaka circular correlation
#
# See Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular
# Statistics. World Scientific, p. 176
immutable JammalamadakaR{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
JammalamadakaR() = JammalamadakaR{false}()
Base.eltype{T<:Real}(::JammalamadakaR, X::AbstractArray{Complex{T}}) = T

function sinmeanphasediff!(out, X)
    for j = 1:size(X, 2)
        # Compute mean phase
        m = zero(eltype(X))
        @simd for i = 1:size(X, 1)
            @inbounds m += X[i, j]
        end

        # Normalize to unit length
        m /= abs(m)

        # Compute difference from mean phase
        @simd for i = 1:size(X, 1)
          @inbounds out[i, j] = imag(X[i, j]*conj(m))
      end
    end
    out
end

allocwork{T<:Real}(t::JammalamadakaR{true}, X::AbstractVecOrMat{Complex{T}}) =
    Array(T, size(X, 1), size(X, 2))
function computestat!{T<:Real}(t::JammalamadakaR{true}, out::AbstractMatrix{T},
                               work::Matrix{T}, X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work, X)

    # Products of phase differences
    At_mul_A!(out, workX)
    cov2coh!(out, out, Base.AbsFun())
end

immutable JammalamadakaRWorkXY{T<:Real}
    workX::Matrix{T}
    workY::Matrix{T}
    sumworkX::Matrix{T}
    sumworkY::Matrix{T}
end
function allocwork{T<:Real}(t::JammalamadakaR{true}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    JammalamadakaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)),
                            Array(T, size(Y, 1), size(Y, 2)),
                            Array(T, 1, nchannels(X)),
                            Array(T, 1, nchannels(Y)))
end
function computestat!{T<:Real}(t::JammalamadakaR{true}, out::AbstractMatrix{T}, work::JammalamadakaRWorkXY{T},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work.workX, X)
    workY = sinmeanphasediff!(work.workY, Y)

    # Products of phase differences
    At_mul_B!(out, workX, workY)
    cov2coh!(out, workX, workY, work.sumworkX, work.sumworkY, out, Base.AbsFun())
end

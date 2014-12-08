#
# Jammalamadaka circular correlation
#
# See Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular
# Statistics. World Scientific, p. 176
immutable JammalamadakaR <: RealPairwiseStatistic
    normalized::Bool
end
JammalamadakaR() = JammalamadakaR(false)

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

immutable JammalamadakaRWorkX{T<:Real}
    workX::Matrix{T}
    normalizedX::Matrix{Complex{T}}

    JammalamadakaRWorkX(w1, w2) = new(w1, w2)
    JammalamadakaRWorkX(w1, w2, w3) = new(w1, w2, w3)
end
function allocwork{T<:Real}(t::JammalamadakaR, X::AbstractVecOrMat{Complex{T}})
    if t.normalized
        JammalamadakaRWorkX{T}(Array(T, size(X, 1), size(X, 2)))
    else
        JammalamadakaRWorkX{T}(Array(T, size(X, 1), size(X, 2)),
                               Array(Complex{T}, size(X, 1), size(X, 2)))
    end
end
function computestat!{T<:Real}(t::JammalamadakaR, out::AbstractMatrix{T},
                               work::JammalamadakaRWorkX{T},
                               X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work.workX, normalized(t, work, X))

    # Products of phase differences
    At_mul_A!(out, workX)
    cov2coh!(out, out, Base.AbsFun())
end

immutable JammalamadakaRWorkXY{T<:Real}
    workX::Matrix{T}
    workY::Matrix{T}
    sumworkX::Matrix{T}
    sumworkY::Matrix{T}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    JammalamadakaRWorkXY(w1, w2, w3, w4) = new(w1, w2, w3, w4)
    JammalamadakaRWorkXY(w1, w2, w3, w4, w5, w6) = new(w1, w2, w3, w4, w5, w6)
end
function allocwork{T<:Real}(t::JammalamadakaR, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    if t.normalized
        JammalamadakaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)),
                                Array(T, size(Y, 1), size(Y, 2)),
                                Array(T, 1, nchannels(X)),
                                Array(T, 1, nchannels(Y)))
    else
        JammalamadakaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)),
                                Array(T, size(Y, 1), size(Y, 2)),
                                Array(T, 1, nchannels(X)),
                                Array(T, 1, nchannels(Y)),
                                Array(Complex{T}, size(X, 1), size(X, 2)),
                                Array(Complex{T}, size(Y, 1), size(Y, 2)))
    end
end
function computestat!{T<:Real}(t::JammalamadakaR, out::AbstractMatrix{T},
                               work::JammalamadakaRWorkXY{T},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    X, Y = normalized(t, work, X, Y)

    # Sins minus mean phases
    workX = sinmeanphasediff!(work.workX, X)
    workY = sinmeanphasediff!(work.workY, Y)

    # Products of phase differences
    At_mul_B!(out, workX, workY)
    cov2coh!(out, workX, workY, work.sumworkX, work.sumworkY, out, Base.AbsFun())
end

#
# Jammalamadaka circular correlation
#
# See Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular
# Statistics. World Scientific, p. 176
immutable JammalamadakaR <: RealPairwiseStatistic
    normalized::Bool
end
JammalamadakaR() = JammalamadakaR(false)

function sins_minus_means!(sins, meanwork, X)
    sum!(meanwork, X)
    unitnormalize!(meanwork, meanwork)
    for j = 1:size(X, 2), i = 1:size(X, 1)
        @inbounds sins[i, j] = imag(X[i, j]*conj(meanwork[i]))
    end
    sins
end

immutable JammalamadakaRWorkX{T<:Real}
    meanphaseX::Matrix{Complex{T}}
    workX::Matrix{T}
    normalizedX::Matrix{Complex{T}}

    JammalamadakaRWorkX(w1, w2) = new(w1, w2)
    JammalamadakaRWorkX(w1, w2, w3) = new(w1, w2, w3)
end
function allocwork{T<:Real}(t::JammalamadakaR, X::AbstractMatrix{Complex{T}})
    if t.normalized
        JammalamadakaRWorkX{T}(Array(Complex{T}, size(X, 1), 1), Array(T, size(X)))
    else
        JammalamadakaRWorkX{T}(Array(Complex{T}, size(X, 1), 1),
                                     Array(T, size(X)),
                                     Array(Complex{T}, size(X)))
    end
end
function computestat!{T<:Real}(t::JammalamadakaR, out::AbstractMatrix{T},
                               work::JammalamadakaRWorkX{T},
                               X::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))

    # Sins minus mean phases
    workX = sins_minus_means!(work.workX, work.meanphaseX, normalized(t, work, X))

    # Products of phase differences
    A_mul_At!(out, workX)
    cov2coh!(out, out, Base.AbsFun())
end

immutable JammalamadakaRWorkXY{T<:Real}
    meanphaseX::Matrix{Complex{T}}
    meanphaseY::Matrix{Complex{T}}
    workX::Matrix{T}
    workY::Matrix{T}
    sumworkX::Vector{T}
    sumworkY::Vector{T}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    JammalamadakaRWorkXY(w1, w2, w3, w4, w5, w6) = new(w1, w2, w3, w4, w5, w6)
    JammalamadakaRWorkXY(w1, w2, w3, w4, w5, w6, w7, w8) = new(w1, w2, w3, w4, w5, w6, w7, w8)
end
function allocwork{T<:Real}(t::JammalamadakaR, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    if t.normalized
        JammalamadakaRWorkXY{T}(Array(Complex{T}, size(X, 1), 1),
                                      Array(Complex{T}, size(Y, 1), 1),
                                      Array(T, size(X)),
                                      Array(T, size(Y)),
                                      Array(T, size(X, 1)),
                                      Array(T, size(Y, 1)))
    else
        JammalamadakaRWorkXY{T}(Array(Complex{T}, size(X, 1), 1),
                                      Array(Complex{T}, size(Y, 1), 1),
                                      Array(T, size(X)),
                                      Array(T, size(Y)),
                                      Array(T, size(X, 1)),
                                      Array(T, size(Y, 1)),
                                      Array(Complex{T}, size(X)),
                                      Array(Complex{T}, size(Y)))
    end
end
function computestat!{T<:Real}(t::JammalamadakaR, out::AbstractMatrix{T},
                               work::JammalamadakaRWorkXY{T},
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    X, Y = normalized(t, work, X, Y)

    # Sins minus mean phases
    workX = sins_minus_means!(work.workX, work.meanphaseX, X)
    workY = sins_minus_means!(work.workY, work.meanphaseY, Y)

    # Products of phase differences
    A_mul_Bt!(out, workX, workY)
    cov2coh!(out, workX, workY, work.sumworkX, work.sumworkY, out, Base.AbsFun())
end
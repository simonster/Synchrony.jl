#
# Circular correlation statistic of Jupp and Mardia
#
# See Jupp, P. E., & Mardia, K. V. (1980). A General Correlation
# Coefficient for Directional Data and Related Regression Problems.
# Biometrika, 67(1), 163–173. doi:10.2307/2335329
#
# The formula in terms of correlation coefficients given in the text
# is slightly wrong. The correct formula is given in:
#
# Mardia, K. V., & Jupp, P. E. (2009). Directional Statistics. John
# Wiley & Sons, p. 249. 
immutable JuppMardiaR <: RealPairwiseStatistic
    normalized::Bool
end
JuppMardiaR() = JuppMardiaR(false)

function meandiff!(out, X)
    for j = 1:size(X, 2)
        # Compute mean resultant
        m = zero(eltype(X))
        @simd for i = 1:size(X, 1)
            @inbounds m += X[i, j]
        end
        m /= size(X, 1)

        # Compute difference from mean
        @simd for i = 1:size(X, 1)
            @inbounds z = X[i, j] - m
            @inbounds out[i, 2j-1] = real(z)
            @inbounds out[i, 2j] = imag(z)
        end
    end
    out
end

immutable JuppMardiaRWorkX{T<:Real}
    workX::Matrix{T}
    covwork::Matrix{T}
    normalizedX::Matrix{Complex{T}}

    JuppMardiaRWorkX(w1, w2) = new(w1, w2)
    JuppMardiaRWorkX(w1, w2, w3) = new(w1, w2, w3)
end
function allocwork{T<:Real}(t::JuppMardiaR, X::AbstractVecOrMat{Complex{T}})
    if t.normalized
        JuppMardiaRWorkX{T}(Array(T, size(X, 1), size(X, 2)*2), 
                            Array(T, size(X, 1)*2, size(X, 1)*2))
    else
        JuppMardiaRWorkX{T}(Array(T, size(X, 1), size(X, 2)*2),
                            Array(T, size(X, 2)*2, size(X, 2)*2),
                            Array(Complex{T}, size(X, 1), size(X, 2)))
    end
end
function computestat!{T<:Real}(t::JuppMardiaR, out::AbstractMatrix{T},
                               work::JuppMardiaRWorkX{T},
                               X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    X = normalized(t, work, X)

    # Compute mean phases
    workX = meandiff!(work.workX, X)

    # Correlation between all components
    r = At_mul_A!(work.covwork, workX)
    cov2coh!(r)

    for j = 1:nchannels(X)
        c1 = 2j-1
        s1 = 2j
        for i = 1:j-1
            c2 = 2i-1
            s2 = 2i
            @inbounds out[i, j] = ((r[c2, c1]^2 + r[s2, c1]^2 + r[c2, s1]^2 + r[s2, s1]^2) +
                                   2*(r[c2, c1]*r[s2, s1] + r[s2, c1]*r[c2, s1])*r[c1, s1]*r[c2, s2] -
                                   2*(r[c2, c1]*r[s2, c1] + r[c2, s1]*r[s2, s1])*r[c2, s2] -
                                   2*(r[c2, c1]*r[c2, s1] + r[s2, c1]*r[s2, s1])*r[c1, s1]) / 
                                  ((1-r[c1, s1]^2)*(1-r[c2, s2]^2))
        end
        @inbounds out[j, j] = 1
    end
    out
end

immutable JuppMardiaRWorkXY{T<:Real}
    workX::Matrix{T}
    workY::Matrix{T}
    covwork::Matrix{T}
    invsqrtsumX::Matrix{T}
    invsqrtsumY::Matrix{T}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    JuppMardiaRWorkXY(w1, w2, w3, w4, w5) = new(w1, w2, w3, w4, w5)
    JuppMardiaRWorkXY(w1, w2, w3, w4, w5, w6, w7) = new(w1, w2, w3, w4, w5, w6, w7)
end
function allocwork{T<:Real}(t::JuppMardiaR, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    if t.normalized
        JuppMardiaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)*2),
                             Array(T, size(Y, 1), size(Y, 2)*2),
                             Array(T, size(X, 2)*2, size(Y, 2)*2),
                             Array(T, size(X, 2)*2),
                             Array(T, size(Y, 2)*2))
    else
        JuppMardiaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)*2),
                             Array(T, size(Y, 1), size(Y, 2)*2),
                             Array(T, size(X, 2)*2, size(Y, 2)*2),
                             Array(T, 1, size(X, 2)*2),
                             Array(T, 1, size(Y, 2)*2),
                             Array(Complex{T}, size(X, 1), size(X, 2)),
                             Array(Complex{T}, size(X, 1), size(Y, 2)))
    end
end
function reimcor!(workX, invsqrtsumX, n)
    for j = 1:2:size(workX, 2)
        v = zero(eltype(workX))
        @simd for i = 1:size(workX, 1)
            @inbounds v += workX[i, j] * workX[i, j+1]
        end
        @inbounds workX[j] = v * invsqrtsumX[j] * invsqrtsumX[j+1]
    end
    workX
end
function computestat!{T<:Real}(t::JuppMardiaR, out::AbstractMatrix{T},
                               work::JuppMardiaRWorkXY{T},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    X, Y = normalized(t, work, X, Y)

    # Compute mean phases
    workX = meandiff!(work.workX, X)
    workY = meandiff!(work.workY, Y)

    # Correlation between signals
    r = At_mul_B!(work.covwork, workX, workY)
    cov2coh!(r, workX, workY, work.invsqrtsumX, work.invsqrtsumY, r)

    # Correlation between real and imaginary components of each signal
    reimcor!(workX, work.invsqrtsumX, size(X, 1))
    reimcor!(workY, work.invsqrtsumY, size(Y, 1))

    for j = 1:nchannels(Y)
        c1 = 2j-1
        s1 = 2j
        for i = 1:nchannels(X)
            c2 = 2i-1
            s2 = 2i
            @inbounds out[i, j] = ((r[c2, c1]^2 + r[s2, c1]^2 + r[c2, s1]^2 + r[s2, s1]^2) +
                                   2*(r[c2, c1]*r[s2, s1] + r[s2, c1]*r[c2, s1])*workY[j]*workX[j] -
                                   2*(r[c2, c1]*r[s2, c1] + r[c2, s1]*r[s2, s1])*workX[j] -
                                   2*(r[c2, c1]*r[c2, s1] + r[s2, c1]*r[s2, s1])*workY[j]) / 
                                  ((1-workY[j]^2)*(1-workX[j]^2))
        end
    end
    out
end

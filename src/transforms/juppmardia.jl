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

function components_minus_means!(out, meanphasework, X)
    sum!(meanphasework, X)
    scale!(meanphasework, 1/size(X, 2))
    
    for j = 1:size(X, 2), i = 1:size(X, 1)
        @inbounds v = X[i, j] - meanphasework[i]
        @inbounds out[2i-1, j] = real(v)
        @inbounds out[2i, j] = imag(v)
    end
    out
end

immutable JuppMardiaRWorkX{T<:Real}
    meanphaseX::Matrix{Complex{T}}
    workX::Matrix{T}
    covwork::Matrix{T}
    normalizedX::Matrix{Complex{T}}

    JuppMardiaRWorkX(w1, w2, w3) = new(w1, w2, w3)
    JuppMardiaRWorkX(w1, w2, w3, w4) = new(w1, w2, w3, w4)
end
function allocwork{T<:Real}(t::JuppMardiaR, X::AbstractMatrix{Complex{T}})
    if t.normalized
        JuppMardiaRWorkX{T}(Array(Complex{T}, size(X, 1), 1),
                                      Array(T, size(X, 1)*2, size(X, 2)), 
                                      Array(T, size(X, 1)*2, size(X, 1)*2))
    else
        JuppMardiaRWorkX{T}(Array(Complex{T}, size(X, 1), 1),
                                      Array(T, size(X, 1)*2, size(X, 2)),
                                      Array(T, size(X, 1)*2, size(X, 1)*2),
                                      Array(Complex{T}, size(X, 1), size(X, 2)))
    end
end
function computestat!{T<:Real}(t::JuppMardiaR, out::AbstractMatrix{T},
                               work::JuppMardiaRWorkX{T},
                               X::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    X = normalized(t, work, X)

    # Compute mean phases
    workX = components_minus_means!(work.workX, work.meanphaseX, X)

    # Correlation between all components
    r = A_mul_At!(work.covwork, workX)
    cov2coh!(r)

    for j = 1:size(X, 1)
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
    meanphaseX::Matrix{Complex{T}}
    meanphaseY::Matrix{Complex{T}}
    workX::Matrix{T}
    workY::Matrix{T}
    covwork::Matrix{T}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    JuppMardiaRWorkXY(w1, w2, w3, w4, w5) = new(w1, w2, w3, w4, w5)
    JuppMardiaRWorkXY(w1, w2, w3, w4, w5, w6, w7) = new(w1, w2, w3, w4, w5, w6, w7)
end
function allocwork{T<:Real}(t::JuppMardiaR, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    if t.normalized
        JuppMardiaRWorkXY{T}(Array(Complex{T}, size(X, 1), 1),
                             Array(Complex{T}, size(Y, 1), 1),
                             Array(T, size(X, 1)*2, size(X, 2)),
                             Array(T, size(X, 1)*2, size(Y, 2)),
                             Array(T, size(X, 1)*2, size(Y, 1)*2))
    else
        JuppMardiaRWorkXY{T}(Array(Complex{T}, size(X, 1), 1),
                             Array(Complex{T}, size(Y, 1), 1),
                             Array(T, size(X, 1)*2, size(X, 2)),
                             Array(T, size(Y, 1)*2, size(Y, 2)),
                             Array(T, size(X, 1)*2, size(Y, 1)*2),
                             Array(Complex{T}, size(X, 1), size(X, 2)),
                             Array(Complex{T}, size(X, 1), size(Y, 2)))
    end
end
function reimcor!(workX, invsqrtsumX, n)
    for i = 1:n
        @inbounds workX[i] = workX[2i-1] * workX[2i]
    end
    for j = 2:size(workX, 2), i = 1:n
        @inbounds workX[i] += workX[2i-1, j] * workX[2i, j]
    end
    for i = 1:n
        @inbounds workX[i] *= invsqrtsumX[2i-1]*invsqrtsumX[2i]
    end
end
function computestat!{T<:Real}(t::JuppMardiaR, out::AbstractMatrix{T},
                               work::JuppMardiaRWorkXY{T},
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    size(out, 1) == size(out, 2) == size(X, 1) || error(DimensionMismatch("out"))
    X, Y = normalized(t, work, X, Y)

    # Compute mean phases
    meanphaseX = work.meanphaseX
    workX = components_minus_means!(work.workX, meanphaseX, X)
    meanphaseY = work.meanphaseY
    workY = components_minus_means!(work.workY, meanphaseY, Y)

    # Correlation between signals
    r = A_mul_Bt!(work.covwork, workX, workY)
    invsqrtsumX = reinterpret(T, meanphaseX, (2*size(X, 1),))
    invsqrtsumY = reinterpret(T, meanphaseY, (2*size(Y, 1),))
    cov2coh!(r, workX, workY, invsqrtsumX, invsqrtsumY, r)

    # Correlation between real and imaginary components of each signal
    reimcor!(workX, invsqrtsumX, size(X, 1))
    reimcor!(workY, invsqrtsumY, size(Y, 1))

    for j = 1:size(Y, 1)
        c1 = 2j-1
        s1 = 2j
        for i = 1:size(X, 1)
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

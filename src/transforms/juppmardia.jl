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
immutable JuppMardiaR{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
JuppMardiaR() = JuppMardiaR{false}()
Base.eltype{T<:Real}(::JuppMardiaR, X::AbstractArray{Complex{T}}) = T

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

    JuppMardiaRWorkX(w1, w2) = new(w1, w2)
    JuppMardiaRWorkX(w1, w2, w3) = new(w1, w2, w3)
end
allocwork{T<:Real}(t::JuppMardiaR{true}, X::AbstractVecOrMat{Complex{T}}) =
    (Array(T, size(X, 1), size(X, 2)*2),  Array(T, size(X, 2)*2, size(X, 2)*2))
function computestat!{T<:Real}(t::JuppMardiaR{true}, out::AbstractMatrix{T}, work::Tuple{Matrix{T}, Matrix{T}},
                               X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    workX, covwork = work

    # Correlation between all components
    r = At_mul_A!(covwork, meandiff!(workX, X))
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
        @inbounds out[j, j] = 2
    end
    out
end

immutable JuppMardiaRWorkXY{T<:Real}
    workX::Matrix{T}
    workY::Matrix{T}
    covwork::Matrix{T}
    invsqrtsumX::Matrix{T}
    invsqrtsumY::Matrix{T}
end
function allocwork{T<:Real}(t::JuppMardiaR{true}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    JuppMardiaRWorkXY{T}(Array(T, size(X, 1), size(X, 2)*2),
                         Array(T, size(Y, 1), size(Y, 2)*2),
                         Array(T, size(X, 2)*2, size(Y, 2)*2),
                         Array(T, 1, size(X, 2)*2),
                         Array(T, 1, size(Y, 2)*2))
end
function reimcor!(workX, invsqrtsumX, n)
    for j = 1:div(size(workX, 2), 2)
        v = zero(eltype(workX))
        @simd for i = 1:size(workX, 1)
            @inbounds v += workX[i, 2j-1] * workX[i, 2j]
        end
        @inbounds workX[j] = v * invsqrtsumX[2j-1] * invsqrtsumX[2j]
    end
    workX
end
function computestat!{T<:Real}(t::JuppMardiaR{true}, out::AbstractMatrix{T}, work::JuppMardiaRWorkXY{T},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)

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
                                   2*(r[c2, c1]*r[s2, s1] + r[s2, c1]*r[c2, s1])*workY[j]*workX[i] -
                                   2*(r[c2, c1]*r[s2, c1] + r[c2, s1]*r[s2, s1])*workX[i] -
                                   2*(r[c2, c1]*r[c2, s1] + r[s2, c1]*r[s2, s1])*workY[j]) / 
                                  ((1-workY[j]^2)*(1-workX[i]^2))
        end
    end
    out
end

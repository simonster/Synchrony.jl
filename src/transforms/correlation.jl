#
# Coherence (as the square root of the correlation matrix)
#

immutable Correlation <: PairwiseStatistic; end
Base.eltype{T<:Real}(::Correlation, X::AbstractArray{T}) = T

# Single input matrix
allocwork{T<:Real}(::Correlation, X::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), Array(T, 1, size(X, 2)))
function computestat!{T<:Real}(::Correlation, out::AbstractMatrix{T}, work::Tuple{Matrix{T},Matrix{T}},
                               X::AbstractVecOrMat{T})
    Xmμ, μ = work
    mean!(μ, X)
    broadcast!(-, Xmμ, X, μ)
    cov2coh!(out, Ac_mul_A!(out, Xmμ))
end

# Two input matrices
allocwork{T<:Real}(::Correlation, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), Array(T, size(Y, 1), size(Y, 2)), cov2coh_work(X), cov2coh_work(Y))
function computestat!{T<:Real}(::Correlation, out::AbstractMatrix{T},
                               work::Tuple{Matrix{T}, Matrix{T}, Array{T}, Array{T}},
                               X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    Xmμ, Ymμ, Xμ, Yμ = work
    mean!(Xμ, X)
    broadcast!(-, Xmμ, X, Xμ)
    mean!(Yμ, Y)
    broadcast!(-, Ymμ, Y, Yμ)
    cov2coh!(out, Xmμ, Ymμ, Xμ, Yμ, Ac_mul_B!(out, Xmμ, Ymμ))
end

#
# Jackknifing for Correlation
#
surrogateval(::Correlation, v) = v

# Single input matrix
allocwork{T<:Real}(t::JackknifeSurrogates{Correlation},
                   X::AbstractVecOrMat{T}) = (Array(T, size(X, 1), size(X, 2)),
                                              Array(T, 1, size(X, 2)),
                                              Array(T, div(size(X, 1), jnn(t)), size(X, 2)))
function computestat!{T<:Real}(t::JackknifeSurrogates{Correlation},
                               out::JackknifeSurrogatesOutput,
                               work::NTuple{3,Matrix{T}},
                               X::AbstractVecOrMat{T})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    Xmμ, μ, jninvmoment = work
    chkinput(trueval, X)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    # Compute means and comoments
    mean!(μ, X)
    broadcast!(-, Xmμ, X, μ)
    Ac_mul_A!(trueval, Xmμ)

    # Compute jackknifed moments
    multiplier = ntrials(X)/(ntrials(X) - 1)
    @inbounds for j = 1:size(X, 2)
        ssq = trueval[j, j]
        @simd for i = 1:size(surrogates, 1)
            v = ssq - abs2(Xmμ[i, j])*multiplier
            jninvmoment[i, j] = sqrt(1/ifelse(v < 0, zero(v), v))
        end
    end

    # Surrogates
    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1
            comoment = trueval[j, k]
            @simd for i = 1:size(surrogates, 1)
                surrogates[i, j, k] = (comoment - Xmμ[i, j]*Xmμ[i, k]*multiplier)*(jninvmoment[i, j]*jninvmoment[i, k])
            end
        end
        for i = 1:size(surrogates, 1)
            surrogates[i, k, k] = 1
        end
    end

    # Finish true value
    cov2coh!(trueval, trueval)

    out
end

# Two input matrices
allocwork{T<:Real}(t::JackknifeSurrogates{Correlation}, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), Array(T, size(Y, 1), size(Y, 2)),
     Array(T, 1, size(X, 2)), Array(T, 1, size(Y, 2)),
     Array(T, div(size(X, 1), jnn(t)), size(X, 2)),
     Array(T, div(size(Y, 1), jnn(t)), size(Y, 2)))
function computestat!{T<:Real}(t::JackknifeSurrogates{Correlation},
                               out::JackknifeSurrogatesOutput,
                               work::NTuple{6,Matrix{T}},
                               X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    Xmμ, Ymμ, Xμ, Yμ, Xjninvmoment, Yjninvmoment = work
    chkinput(trueval, X, Y)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    # Compute means and comoments
    mean!(Xμ, X)
    broadcast!(-, Xmμ, X, Xμ)
    mean!(Yμ, Y)
    broadcast!(-, Ymμ, Y, Yμ)
    Ac_mul_B!(trueval, Xmμ, Ymμ)

    # Compute jackknifed moments
    multiplier = ntrials(X)/(ntrials(X) - 1)
    for (arrmμ, jninvmoment) in ((Xmμ, Xjninvmoment), (Ymμ, Yjninvmoment))
        @inbounds for j = 1:size(arrmμ, 2)
            ssq = zero(T)
            @simd for i = 1:size(arrmμ, 1)
                ssq += abs2(arrmμ[i, j])
            end
            @simd for i = 1:size(surrogates, 1)
                jninvmoment[i, j] = sqrt(1/(ssq - abs2(arrmμ[i, j])*multiplier))
            end
        end
    end

    # Surrogates
    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2)
        comoment = trueval[j, k]
        @simd for i = 1:size(surrogates, 1)
            surrogates[i, j, k] = (comoment - Ymμ[i, k]*Xmμ[i, j]*multiplier)*(Yjninvmoment[i, k]*Xjninvmoment[i, j])
        end
    end

    # Finish true value
    cov2coh!(trueval, Xmμ, Ymμ, Xμ, Yμ, trueval)

    out
end

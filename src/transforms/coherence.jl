#
# Coherency
#

immutable Coherency <: ComplexPairwiseStatistic; end

# Single input matrix
allocwork{T<:Real}(::Coherency, X::AbstractVecOrMat{Complex{T}}) = nothing
computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}}, work::Nothing,
                      X::AbstractVecOrMat{Complex{T}}) = 
    cov2coh!(out, Ac_mul_A!(out, X))

# Two input matrices
allocwork{T<:Real}(::Coherency, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(T, 1, nchannels(X)), Array(T, 1, nchannels(Y)))
function computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}},
                      work::(Matrix{T}, Matrix{T}),
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    cov2coh!(out, X, Y, work[1], work[2], Ac_mul_B!(out, X, Y))
end

#
# Coherence (as the square root of the correlation matrix)
#

immutable Coherence <: RealPairwiseStatistic; end

# Single input matrix
allocwork{T<:Real}(::Coherence, X::AbstractVecOrMat{Complex{T}}) =
    Array(Complex{T}, nchannels(X), nchannels(X))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::Matrix{Complex{T}},
                      X::AbstractVecOrMat{Complex{T}}) =
    cov2coh!(out, Ac_mul_A!(work, X), Base.AbsFun())

# Two input matrices
allocwork{T<:Real}(::Coherence, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), Array(T, nchannels(X), 1), Array(T, nchannels(Y), 1))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::(Matrix{Complex{T}}, Matrix{T}, Matrix{T}),
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) = 
    cov2coh!(out, X, Y, work[2], work[3], Ac_mul_B!(work[1], X, Y), Base.AbsFun())

#
# Jackknifing for Coherency/Coherence
#
surrogateval(::Jackknife{Coherence}, v) = abs(v)
surrogateval(::Jackknife{Coherency}, v) = v

# Single input matrix
allocwork{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                   X::AbstractVecOrMat{Complex{T}}) = allocwork(t.transform, X)
function computestat!{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                               out::JackknifeOutput,
                               work::Union(Matrix{Complex{T}}, Nothing),
                               X::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc::Matrix{Complex{T}} = isa(t, Jackknife{Coherence}) ? work : trueval
    chkinput(trueval, X)

    Ac_mul_A!(XXc, X)

    # Surrogates
    @inbounds for k = 1:size(X, 2)
        kssq = real(XXc[k, k])
        for j = 1:k-1
            jssq = real(XXc[j, j])
            for i = 1:size(X, 1)
                v = XXc[j, k] - conj(X[i, j])*X[i, k]
                # XXX maybe precompute sqrt for each channel and trial?
                surrogates[i, j, k] = surrogateval(t, v)/sqrt((jssq - abs2(X[i, j]))*(kssq - abs2(X[i, k])))
            end
        end
        surrogates[:, k, k] = one(eltype(surrogates))
    end

    # Finish true value
    cov2coh!(trueval, XXc, isa(t, Jackknife{Coherence}) ? Base.AbsFun() : Base.IdFun())

    out
end

# Two input matrices
allocwork{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                   X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (allocwork(t.transform, X, Y), Array(T, nchannels(X)), Array(T, nchannels(Y)))
function computestat!{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                               out::JackknifeOutput,
                               work::(Union(Matrix{Complex{T}}, Nothing), Vector{T}, Vector{T}),
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XYc::Matrix{Complex{T}} = isa(t, Jackknife{Coherence}) ? work[1] : trueval
    chkinput(trueval, X, Y)

    Ac_mul_B!(XYc, X, Y)

    # Surrogates
    @inbounds for k = 1:size(Y, 2)
        kssq = zero(T)
        for i = 1:size(X, 1)
            kssq += abs2(Y[i, k])
        end

        for j = 1:size(X, 2)
            jssq = zero(T)
            for i = 1:size(X, 1)
                jssq += abs2(X[i, j])
            end

            for i = 1:size(X, 1)
                v = XXc[j, k] - conj(X[i, j])*Y[i, k]
                # XXX maybe precompute sqrt for each channel and trial?
                surrogates[i, j, k] = surrogateval(t, v)/sqrt((jssq - abs2(X[i, j]))*(kssq - abs2(Y[i, k])))
            end
        end
    end

    # Finish true value
    cov2coh!(trueval, X, Y, work[2], work[3], XYc, isa(t, Jackknife{Coherence}) ? Base.AbsFun() : Base.IdFun())

    out
end

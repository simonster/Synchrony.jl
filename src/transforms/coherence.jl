#
# Coherency
#

immutable Coherency <: PairwiseStatistic; end
Base.eltype{T<:Real}(::Coherency, X::AbstractArray{Complex{T}}) = Complex{T}

# Single input matrix
allocwork{T<:Real}(::Coherency, X::AbstractVecOrMat{Complex{T}}) = nothing
computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}}, work::Void,
                      X::AbstractVecOrMat{Complex{T}}) = 
    cov2coh!(out, Ac_mul_A!(out, X))

# Two input matrices
allocwork{T<:Real}(::Coherency, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (cov2coh_work(X), cov2coh_work(Y))
function computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}},
                      work::Tuple{Array{T}, Array{T}},
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    cov2coh!(out, X, Y, work[1], work[2], Ac_mul_B!(out, X, Y))
end

#
# Coherence (as the square root of the correlation matrix)
#

immutable Coherence <: PairwiseStatistic; end
Base.eltype{T<:Real}(::Coherence, X::AbstractArray{Complex{T}}) = T

# Single input matrix
allocwork{T<:Real}(::Coherence, X::AbstractVecOrMat{Complex{T}}) =
    Array(Complex{T}, nchannels(X), nchannels(X))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::Matrix{Complex{T}},
                      X::AbstractVecOrMat{Complex{T}}) =
    cov2coh!(out, Ac_mul_A!(work, X), Base.AbsFun())

# Two input matrices
allocwork{T<:Real}(::Coherence, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::Tuple{Matrix{Complex{T}}, Array{T}, Array{T}},
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) = 
    cov2coh!(out, X, Y, work[2], work[3], Ac_mul_B!(work[1], X, Y), Base.AbsFun())

#
# Jackknifing for Coherency/Coherence
#
surrogateval(::Coherence, v) = abs(v)
surrogateval(::Coherency, v) = v

# Single input matrix
allocwork{T<:Real}(t::Union(AbstractJackknifeSurrogates{Coherency}, AbstractJackknifeSurrogates{Coherence}),
                   X::AbstractVecOrMat{Complex{T}}) = allocwork(t.transform, X)
accumulator_array(::AbstractJackknifeSurrogates{Coherency}, work::Nothing, out::AbstractMatrix) = out
accumulator_array(::AbstractJackknifeSurrogates{Coherence}, work::AbstractMatrix, out::AbstractMatrix) = work
function computestat!{T<:Real}(t::Union(AbstractJackknifeSurrogates{Coherency}, AbstractJackknifeSurrogates{Coherence}),
                               out::JackknifeSurrogatesOutput,
                               work::Union(Matrix{Complex{T}}, Nothing),
                               X::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc = accumulator_array(t, work, out.trueval)
    chkinput(trueval, X)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    Ac_mul_A!(XXc, X)

    # Surrogates
    @inbounds for k = 1:size(X, 2)
        kssq = real(XXc[k, k])
        for j = 1:k-1
            jssq = real(XXc[j, j])
            for i = 1:size(surrogates, 1)
                v = XXc[j, k]
                jssqdel = jssq
                kssqdel = kssq
                for idel = (i-1)*jnn(t)+1:i*jnn(t)
                    v -= conj(X[idel, j])*X[idel, k]
                    jssqdel -= abs2(X[idel, j])
                    kssqdel -= abs2(X[idel, k])
                end
                # XXX maybe precompute sqrt for each channel and trial?
                surrogates[i, j, k] = surrogateval(t.transform, v)/sqrt(jssqdel*kssqdel)
            end
        end
        for i = 1:size(surrogates, 1)
            surrogates[i, k, k] = 1
        end
    end

    # Finish true value
    if isa(t.transform, Coherence)
        cov2coh!(trueval, XXc, Base.AbsFun())
    else
        cov2coh!(trueval, XXc, Base.IdFun())
    end

    out
end

# Two input matrices
allocwork{T<:Real}(t::JackknifeSurrogates{Coherency}, X::AbstractVecOrMat{Complex{T}},
                   Y::AbstractVecOrMat{Complex{T}}) =
    (nothing, cov2coh_work(X), cov2coh_work(Y))
allocwork{T<:Real}(t::JackknifeSurrogates{Coherence}, X::AbstractVecOrMat{Complex{T}},
                   Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y))
function computestat!{T<:Real,V}(t::Union(JackknifeSurrogates{Coherency}, JackknifeSurrogates{Coherence}),
                                 out::JackknifeSurrogatesOutput,
                                 work::Tuple{V, Array{T}, Array{T}},
                                 X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XYc = accumulator_array(t, work[1], out.trueval)
    chkinput(trueval, X, Y)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

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

            for i = 1:div(size(X, 1), jnn(t))
                v = XYc[j, k]
                jssqdel = jssq
                kssqdel = kssq
                for idel = (i-1)*jnn(t)+1:i*jnn(t)
                    v -= conj(X[idel, j])*Y[idel, k]
                    jssqdel -= abs2(X[idel, j])
                    kssqdel -= abs2(Y[idel, k])
                end
                # XXX maybe precompute sqrt for each channel and trial?
                surrogates[i, j, k] = surrogateval(t.transform, v)/sqrt(jssqdel*kssqdel)
            end
        end
    end

    # Finish true value
    if isa(t, JackknifeSurrogates{Coherence})
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.AbsFun())
    else
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.IdFun())
    end

    out
end

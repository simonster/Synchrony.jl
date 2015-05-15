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
    (cov2coh_work(X), cov2coh_work(Y))
function computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}},
                      work::@compat(Tuple{Array{T}, Array{T}}),
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
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::@compat(Tuple{Matrix{Complex{T}}, Array{T}, Array{T}}),
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) = 
    cov2coh!(out, X, Y, work[2], work[3], Ac_mul_B!(work[1], X, Y), Base.AbsFun())

#
# Jackknifing for Coherency/Coherence
#
surrogateval(::Coherence, v) = abs(v)
surrogateval(::Coherency, v) = v

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
                surrogates[i, j, k] = surrogateval(t.transform, v)/sqrt((jssq - abs2(X[i, j]))*(kssq - abs2(X[i, k])))
            end
        end
        for i = 1:size(X, 1)
            surrogates[i, k, k] = 1
        end
    end

    # Finish true value
    if isa(t, Jackknife{Coherence})
        cov2coh!(trueval, XXc, Base.AbsFun())
    else
        cov2coh!(trueval, XXc, Base.IdFun())
    end

    out
end

# Two input matrices
allocwork{T<:Real}(t::Jackknife{Coherency}, X::AbstractVecOrMat{Complex{T}},
                   Y::AbstractVecOrMat{Complex{T}}) =
    (nothing, cov2coh_work(X), cov2coh_work(Y))
allocwork{T<:Real}(t::Jackknife{Coherence}, X::AbstractVecOrMat{Complex{T}},
                   Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, nchannels(X), nchannels(Y)), cov2coh_work(X), cov2coh_work(Y))
function computestat!{T<:Real,V}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                                 out::JackknifeOutput,
                                 work::@compat(Tuple{V, Array{T}, Array{T}}),
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
                v = XYc[j, k] - conj(X[i, j])*Y[i, k]
                # XXX maybe precompute sqrt for each channel and trial?
                surrogates[i, j, k] = surrogateval(t.transform, v)/sqrt((jssq - abs2(X[i, j]))*(kssq - abs2(Y[i, k])))
            end
        end
    end

    # Finish true value
    if isa(t, Jackknife{Coherence})
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.AbsFun())
    else
        cov2coh!(trueval, X, Y, work[2], work[3], XYc, Base.IdFun())
    end

    out
end

#
# Bootstrapping for Coherency/Coherence
#

# Single input matrix
allocwork{T<:Real}(t::Bootstrap{Coherency}, X::AbstractVecOrMat{Complex{T}}) = nothing
allocwork{T<:Real}(t::Bootstrap{Coherence}, X::AbstractVecOrMat{Complex{T}}) =
    Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(X))
function computestat!{S,T<:Real}(t::Union(Bootstrap{Coherency}, Bootstrap{Coherence}),
                                 out::AbstractArray{S,3},
                                 work::Union(Array{Complex{T}, 3}, Nothing), X::AbstractVecOrMat{Complex{T}})
    weights = t.weights
    XXc::Array{Complex{T}, 3} = isa(t, Bootstrap{Coherence}) ? work : out

    size(out, 1) == size(weights, 1) && size(out, 2) == nchannels(X) &&
        size(out, 3) == nchannels(X) || throw(DimensionMismatch("output size mismatch"))
    size(XXc, 1) == size(out, 1) && size(XXc, 2) == size(out, 2) &&
        size(XXc, 3) == size(out, 3) || throw(DimensionMismatch("work size mimatch"))

    fill!(XXc, zero(T))

    # Compute cross-spectrum for each bootstrap
    @inbounds for k = 1:size(X, 2), j = 1:k, i = 1:size(X, 1)
        v = conj(X[i, j])*X[i, k]
        @simd for iboot = 1:size(weights, 1)
            XXc[iboot, j, k] += weights[iboot, i]*v
        end
    end

    # Finish coherence/coherency
    @inbounds for k = 1:size(X, 2)
        for iboot = 1:size(weights, 1)
            out[iboot, k, k] = 1/sqrt(real(XXc[iboot, k, k]))
        end
        for j = 1:k-1, iboot = 1:size(weights, 1)
            out[iboot, j, k] = surrogateval(t.transform, XXc[iboot, j, k])*out[iboot, j, j]*out[iboot, k, k]
        end
    end
    @inbounds for k = 1:size(X, 2), iboot = 1:size(weights, 1)
        out[iboot, k, k] = 1
    end
    out
end

# Two input matrices
allocwork{T<:Real}(t::Bootstrap{Coherency}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (nothing, Array(T, size(t.weights, 1), nchannels(X)), Array(T, size(t.weights, 1), nchannels(Y)))
allocwork{T<:Real}(t::Bootstrap{Coherence}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(Y)),
     Array(T, size(t.weights, 1), nchannels(X)), Array(T, size(t.weights, 1), nchannels(Y)))
function computestat!{S,T<:Real,V}(t::Union(Bootstrap{Coherency}, Bootstrap{Coherence}),
                                   out::AbstractArray{S,3},
                                   work::@compat(Tuple{V, Vector{T}, Vector{T}}),
                                   X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    weights = t.weights
    XYc::Array{Complex{T}, 3} = isa(t, Bootstrap{Coherence}) ? work[1] : out
    psdX = work[2]
    psdY = work[3]

    size(out, 1) == size(weights, 1) && size(out, 2) == nchannels(X) &&
        size(out, 3) == nchannels(X) || throw(DimensionMismatch("output size mismatch"))
    size(XYc, 1) == size(out, 1) && size(XYc, 2) == size(out, 2) &&
        size(XYc, 3) == size(out, 3) && size(psdX, 1) == size(weights, 1) &&
        size(psdX, 2) == nchannels(X) && size(psdY, 1) == size(weights, 1) &&
        size(psdY, 2) == nchannels(Y) || throw(DimensionMismatch("work size mimatch"))

    fill!(XYc, zero(Complex{T}))
    fill!(psdX, zero(T))
    fill!(psdY, zero(T))

    @inbounds for k = 1:size(Y, 2)
        # Compute PSD of X channels for each bootstrap
        for i = 1:size(X, 1)
            v = abs2(Y[i, k])
            @simd for iboot = 1:size(weights, 1)
                psdY[iboot, k] += weights[iboot, i]*v
            end
        end

        # Compute cross-spectrum for each bootstrap
        for j = 1:size(X, 2), i = 1:size(X, 1)
            v = conj(X[i, j])*Y[i, k]
            @simd for iboot = 1:size(weights, 1)
                XYc[iboot, j, k] += weights[iboot, i]*v
            end
        end
    end

    # Compute PSD of Y channels for each bootstrap
    @inbounds for j = 1:size(X, 2), i = 1:size(X, 1)
        v = abs2(X[i, j])
        @simd for iboot = 1:size(weights, 1)
            psdX[iboot, j] += weights[iboot, i]*v
        end
    end

    # Finish coherence/coherency
    for arr in (psdX, psdY)
        @simd for i = 1:length(arr)
            @inbounds arr[i] = 1/sqrt(arr[i])
        end
    end
    for k = 1:size(Y, 2), j = 1:size(X, 2), iboot = 1:size(weights, 1)
        @inbounds out[iboot, j, k] = surrogateval(t.transform, XYc[iboot, j, k])*psdX[iboot, j]*psdY[iboot, k]
    end
    out
end

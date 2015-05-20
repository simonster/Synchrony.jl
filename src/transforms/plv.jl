#
# Circular mean of phase difference
#

immutable MeanPhaseDiff <: ComplexPairwiseStatistic
    normalized::Bool
end
MeanPhaseDiff() = MeanPhaseDiff(false)

immutable MeanPhaseDiffWork{T<:Real}
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    MeanPhaseDiffWork() = new()
    MeanPhaseDiffWork(x) = new(x)
    MeanPhaseDiffWork(x, y) = new(x, y)
end

# Single input matrix
allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() : MeanPhaseDiffWork{T}(Array(Complex{T}, size(X, 1), size(X, 2)))
finish(::MeanPhaseDiff, x, n) = x/n
@normalized function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                                           work::MeanPhaseDiffWork{T}, X::AbstractVecOrMat{Complex{T}})
    scale!(Ac_mul_A!(out, X), 1/ntrials(X))
end

# Two input matrices
allocwork{T<:Real}(t::MeanPhaseDiff, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    t.normalized ? MeanPhaseDiffWork{T}() :
        MeanPhaseDiffWork{T}(Array(Complex{T}, size(X, 1), size(X, 2)), Array(Complex{T}, size(Y, 1), size(Y, 2)))
finish_xy!(::MeanPhaseDiff, out, work, n) = scale!(out, work, 1/n)
@normalized function computestat!{T<:Real}(t::MeanPhaseDiff, out::AbstractMatrix{Complex{T}},
                                          work::MeanPhaseDiffWork{T}, X::AbstractVecOrMat{Complex{T}},
                                          Y::AbstractVecOrMat{Complex{T}})
    scale!(Ac_mul_B!(out, X, Y), 1/ntrials(X))
end

#
# Phase locking value, pairwise phase consistency
#
# For PLV, see Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela,
# F. J. (1999). Measuring phase synchrony in brain signals. Human Brain
# Mapping, 8(4), 194–208.
# doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C
#
# For PPC, see Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P.,
# & Pennartz, C. M. A. (2010). The pairwise phase consistency: A
# bias-free measure of rhythmic neuronal synchronization. NeuroImage,
# 51(1), 112–122. doi:10.1016/j.neuroimage.2010.01.073

immutable PLV <: RealPairwiseStatistic
    normalized::Bool
end
PLV() = PLV(false)

immutable PPC <: RealPairwiseStatistic
    normalized::Bool
end
PPC() = PPC(false)

immutable PLVWork{T<:Real,S}
    meanphasediff::S
    normalizedX::Matrix{Complex{T}}
    normalizedY::Matrix{Complex{T}}

    PLVWork(w1) = new(w1)
    PLVWork(w1, w2) = new(w1, w2)
    PLVWork(w1, w2, w3) = new(w1, w2, w3)
end

accumulator{T<:Real}(::Union(Type{MeanPhaseDiff}, Type{PLV}, Type{PPC}), ::Type{T}) = zero(Complex{T})
@inline accumulate{T<:Real}(::Union(Type{MeanPhaseDiff}, Type{PLV}, Type{PPC}), x::Complex{T},
                            v1::Complex{T}, v2::Complex{T}) = (x + conj(v1)*v2)
@inline accumulate{T<:Real}(::Union(Type{MeanPhaseDiff}, Type{PLV}, Type{PPC}), x::Complex{T},
                            v1::Complex{T}, v2::Complex{T}, weight::Real) = (x + conj(v1)*v2*weight)
finish(::Type{MeanPhaseDiff}, x::Complex, n::Int) = x/n
finish(::Type{PLV}, x::Complex, n::Int) = abs(x)/n
finish(::Type{PPC}, x::Complex, n::Int) = abs2(x)/(n*(n-1)) - 1/(n-1)


# Single input matrix
function allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractVecOrMat{Complex{T}})
    if t.normalized
        PLVWork{T,Matrix{Complex{T}}}(Array(Complex{T}, nchannels(X), nchannels(X)))
    else
        PLVWork{T,Matrix{Complex{T}}}(Array(Complex{T}, nchannels(X), nchannels(X)),
                                      Array(Complex{T}, size(X, 1), size(X, 2)))
    end
end
function finish!{T<:PairwiseStatistic}(::Type{T}, out, work, n)
    for j = 1:size(out, 1), i = 1:j
        @inbounds out[i, j] = finish(T, work[i, j], n)
    end
    out
end
@normalized function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T,Matrix{Complex{T}}},
                                           X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    finish!(typeof(t), out, Ac_mul_A!(work.meanphasediff, X), ntrials(X))
end

# Two input matrices
function allocwork{T<:Real}(t::Union(PLV, PPC), X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    if t.normalized
        PLVWork{T,Matrix{Complex{T}}}(Array(Complex{T}, nchannels(X), nchannels(Y)))
    else
        PLVWork{T,Matrix{Complex{T}}}(Array(Complex{T}, nchannels(X), nchannels(Y)),
                                      Array(Complex{T}, size(X, 1), size(X, 2)),
                                      Array(Complex{T}, size(Y, 1), size(Y, 2)))
    end
end
function finish_xy!{T<:PairwiseStatistic}(::Type{T}, out, work, n)
    for i = 1:length(out)
        @inbounds out[i] = finish(T, work[i], n)
    end
    out
end
@normalized function computestat!{T<:Real}(t::Union(PLV, PPC), out::AbstractMatrix{T}, work::PLVWork{T,Matrix{Complex{T}}},
                                           X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    finish_xy!(typeof(t), out, Ac_mul_B!(work.meanphasediff, X, Y), ntrials(X))
end

#
# Jackknifing for MeanPhaseDiff/PLV/PPC
#

# Single input matrix
allocwork{T<:Real}(t::Union(JackknifeSurrogates{MeanPhaseDiff}, JackknifeSurrogates{PPC}, JackknifeSurrogates{PLV}), X::AbstractVecOrMat{Complex{T}}) =
    allocwork(PLV(t.transform.normalized), X)
@normalized t.transform function computestat!{S<:Union(MeanPhaseDiff, PLV, PPC), T<:Real}(t::JackknifeSurrogates{S},
                                                                                          out::JackknifeSurrogatesOutput,
                                                                                          work::Union(MeanPhaseDiffWork{T}, PLVWork{T,Matrix{Complex{T}}}),
                                                                                          X::AbstractVecOrMat{Complex{T}})
    trueval = out.trueval
    surrogates = out.surrogates
    XXc = work.meanphasediff

    chkinput(trueval, X)

    Ac_mul_A!(XXc, X)

    n = ntrials(X)
    finish!(S, trueval, XXc, n)

    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1
            x = XXc[j, k]
            for i = 1:size(X, 1)
                surrogates[i, j, k] = finish(S, x - conj(X[i, j])*X[i, k], n-1)
            end
        end
        for i = 1:size(X, 1)
            surrogates[i, k, k] = 1
        end
    end

    out
end

# Two input matrices
allocwork{T<:Real}(t::Union(JackknifeSurrogates{MeanPhaseDiff}, JackknifeSurrogates{PPC}, JackknifeSurrogates{PLV}), X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    allocwork(PLV(t.transform.normalized), X, Y)
@normalized t.transform function computestat!{S<:Union(MeanPhaseDiff, PLV, PPC), T<:Real}(t::JackknifeSurrogates{S},
                                                                                          out::JackknifeSurrogatesOutput,
                                                                                          work::Union(MeanPhaseDiffWork{T}, PLVWork{T,Matrix{Complex{T}}}),
                                                                                          X::AbstractVecOrMat{Complex{T}},
                                                                                          Y::AbstractVecOrMat{Complex{T}})
    trueval = out.trueval
    surrogates = out.surrogates
    XYc = work.meanphasediff

    chkinput(trueval, X, Y)

    Ac_mul_B!(XYc, X, Y)

    n = ntrials(X)
    finish_xy!(S, trueval, XYc, n)

    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2)
        x = XYc[j, k]
        for i = 1:size(X, 1)
            surrogates[i, j, k] = finish(S, x - conj(X[i, j])*Y[i, k], n-1)
        end
    end

    out
end

#
# Bootstrapping for PLV/PPC
#

# Single input matrix
allocwork{T<:Real}(t::Bootstrap{MeanPhaseDiff}, X::AbstractVecOrMat{Complex{T}}) = allocwork(t.transform, X)
function allocwork{T<:Real}(t::Union(Bootstrap{PLV}, Bootstrap{PPC}), X::AbstractVecOrMat{Complex{T}})
    if t.transform.normalized
        PLVWork{T,Array{Complex{T},3}}(Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(X)))
    else
        PLVWork{T,Array{Complex{T},3}}(Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(X)),
                                       Array(Complex{T}, size(X, 1), size(X, 2)))
    end
end
@normalized t.transform function computestat!{S<:Union(MeanPhaseDiff,PLV,PPC),U,T<:Real}(t::Bootstrap{S}, out::AbstractArray{U,3},
                                                                                         work::Union(MeanPhaseDiffWork{T}, PLVWork{T,Array{Complex{T},3}}),
                                                                                         X::AbstractVecOrMat{Complex{T}})
    weights = t.weights
    XXc::Array{Complex{T}, 3} = isa(t, Bootstrap{MeanPhaseDiff}) ? out : work.meanphasediff

    size(out, 1) == size(weights, 1) && size(out, 2) == nchannels(X) &&
        size(out, 3) == nchannels(X) || error(DimensionMismatch("output size mismatch"))
    size(XXc, 1) == size(out, 1) && size(XXc, 2) == size(out, 2) &&
        size(XXc, 3) == size(out, 3) || error(DimensionMismatch("work size mimatch"))

    fill!(XXc, zero(T))

    # Compute sum of phase differences for each bootstrap
    @inbounds for k = 1:size(X, 2), j = 1:k-1, i = 1:size(X, 1)
        v1 = X[i, j]
        v2 = X[i, k]
        @simd for iboot = 1:size(weights, 1)
            XXc[iboot, j, k] = accumulate(S, XXc[iboot, j, k], v1, v2, weights[iboot, i])
        end
    end

    # Finish
    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1, i = 1:size(XXc, 1)
            out[i, j, k] = finish(S, XXc[i, j, k], ntrials(X))
        end
        for i = 1:size(XXc, 1)
            out[i, k, k] = 1
        end
    end
    out
end

# Two input matrices
allocwork{T<:Real}(t::Bootstrap{MeanPhaseDiff}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    allocwork(t.transform, X, Y)
function allocwork{T<:Real}(t::Union(Bootstrap{PLV}, Bootstrap{PPC}), X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    if t.transform.normalized
        PLVWork{T,Array{Complex{T},3}}(Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(Y)))
    else
        PLVWork{T,Array{Complex{T},3}}(Array(Complex{T}, size(t.weights, 1), nchannels(X), nchannels(Y)),
                                       Array(Complex{T}, size(X, 1), size(X, 2)),
                                       Array(Complex{T}, size(Y, 1), size(Y, 2)))
    end
end
@normalized t.transform function computestat!{S<:Union(MeanPhaseDiff,PLV,PPC),U,T<:Real}(t::Bootstrap{S}, out::AbstractArray{U,3},
                                                                                         work::Union(MeanPhaseDiffWork{T}, PLVWork{T,Array{Complex{T},3}}),
                                                                                         X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    weights = t.weights
    XYc::Array{Complex{T}, 3} = isa(t, Bootstrap{MeanPhaseDiff}) ? out : work.meanphasediff

    size(out, 1) == size(weights, 1) && size(out, 2) == nchannels(X) &&
        size(out, 3) == nchannels(X) || error(DimensionMismatch("output size mismatch"))
    size(XYc, 1) == size(out, 1) && size(XYc, 2) == size(out, 2) &&
        size(XYc, 3) == size(out, 3) || error(DimensionMismatch("work size mimatch"))

    fill!(XYc, zero(Complex{T}))

    # Compute sum of phase differences for each bootstrap
    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2), i = 1:size(X, 1)
        v1 = X[i, j]
        v2 = Y[i, k]
        @simd for iboot = 1:size(weights, 1)
            XYc[iboot, j, k] = accumulate(S, XYc[iboot, j, k], v1, v2, weights[iboot, i])
        end
    end

    # Finish
    for i = 1:length(out)
        @inbounds out[i] = finish(S, XXc[i, j, k], ntrials(X))
    end
    out
end

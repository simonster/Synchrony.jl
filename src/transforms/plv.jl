#
# Circular mean of phase difference
#

immutable MeanPhaseDiff{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
MeanPhaseDiff() = MeanPhaseDiff{false}()
Base.eltype{T<:Real}(::MeanPhaseDiff, X::AbstractArray{Complex{T}}) = Complex{T}

# Single input matrix
allocwork{T<:Real}(t::MeanPhaseDiff{true}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}=X) = nothing
computestat!{T<:Real}(t::MeanPhaseDiff{true}, out::AbstractMatrix{Complex{T}},
                      work::Void, X::AbstractVecOrMat{Complex{T}}) =
    scale!(Ac_mul_A!(out, X), 1/ntrials(X))

# Two input matrices
finish_xy!(::MeanPhaseDiff, out, work, n) = scale!(out, work, 1/n)
computestat!{T<:Real}(t::MeanPhaseDiff{true}, out::AbstractMatrix{Complex{T}},
                      work::Void, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    scale!(Ac_mul_B!(out, X, Y), 1/ntrials(X))

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

immutable PLV{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
PLV() = PLV{false}()

immutable PPC{Normalized} <: NormalizedPairwiseStatistic{Normalized} end
PPC() = PPC{false}()

Base.eltype{T<:Real}(::Union{PLV, PPC}, X::AbstractArray{Complex{T}}) = T

accumulator{T<:Real}(::Union{Type{MeanPhaseDiff{true}}, Type{PLV{true}}, Type{PPC{true}}}, ::Type{Complex{T}}) = zero(Complex{T})
@inline accumulate{T<:Real}(::Union{Type{MeanPhaseDiff{true}}, Type{PLV{true}}, Type{PPC{true}}}, x::Complex{T},
                            v1::Complex{T}, v2::Complex{T}) = (x + conj(v1)*v2)
@inline accumulate{T<:Real}(::Union{Type{MeanPhaseDiff{true}}, Type{PLV{true}}, Type{PPC{true}}}, x::Complex{T},
                            v1::Complex{T}, v2::Complex{T}, weight::Real) = (x + conj(v1)*v2*weight)
finish(::Type{MeanPhaseDiff{true}}, x::Complex, n::Int) = x/n
finish(::Type{PLV{true}}, x::Complex, n::Int) = abs(x)/n
finish(::Type{PPC{true}}, x::Complex, n::Int) = abs2(x)/(n*(n-1)) - 1/(n-1)
diagval(::Union{Type{MeanPhaseDiff{true}}, Type{PLV{true}}, Type{PPC{true}}}) = 1


# Single input matrix
allocwork{T<:Real}(t::Union{PLV{true}, PPC{true}}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}=X) =
    Array(Complex{T}, nchannels(X), nchannels(Y))
function finish!{T<:PairwiseStatistic}(::Type{T}, out, work, n)
    for j = 1:size(out, 1), i = 1:j
        @inbounds out[i, j] = finish(T, work[i, j], n)
    end
    out
end
function computestat!{T<:Real}(t::Union{PLV{true}, PPC{true}}, out::AbstractMatrix{T}, work::Matrix{Complex{T}},
                               X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    finish!(typeof(t), out, Ac_mul_A!(work, X), ntrials(X))
    for i = 1:size(out, 1)
        @inbounds out[i, i] = 1
    end
    out
end

# Two input matrices
function finish_xy!{T<:PairwiseStatistic}(::Type{T}, out, work, n)
    for i = 1:length(out)
        @inbounds out[i] = finish(T, work[i], n)
    end
    out
end
function computestat!{T<:Real}(t::Union{PLV{true}, PPC{true}}, out::AbstractMatrix{T}, work::Matrix{Complex{T}},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    chkinput(out, X, Y)
    finish_xy!(typeof(t), out, Ac_mul_B!(work, X, Y), ntrials(X))
end

#
# Jackknifing for MeanPhaseDiff/PLV/PPC
#

# Single input matrix
allocwork{T<:Real}(t::Union{AbstractJackknifeSurrogates{MeanPhaseDiff{true}}, AbstractJackknifeSurrogates{PPC{true}}, AbstractJackknifeSurrogates{PLV{true}}},
                   X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}=X) =
    Array(Complex{T}, nchannels(X), nchannels(Y))
function computestat!{S<:Union{MeanPhaseDiff{true}, PLV{true}, PPC{true}}, T<:Real}(t::AbstractJackknifeSurrogates{S},
                                                                                    out::JackknifeSurrogatesOutput,
                                                                                    work::Matrix{Complex{T}},
                                                                                    X::AbstractVecOrMat{Complex{T}})
    trueval = out.trueval
    surrogates = out.surrogates

    chkinput(trueval, X)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    Ac_mul_A!(work, X)

    n = ntrials(X)
    finish!(S, trueval, work, n)

    @inbounds for k = 1:size(X, 2)
        for j = 1:k-1
            x = work[j, k]
            for i = 1:size(surrogates, 1)
                v = x
                for idel = (i-1)*jnn(t)+1:i*jnn(t)
                    v -= conj(X[idel, j])*X[idel, k]
                end
                surrogates[i, j, k] = finish(S, v, n-jnn(t))
            end
        end
        for i = 1:size(surrogates, 1)
            surrogates[i, k, k] = 1
        end
    end

    out
end

# Two input matrices
function computestat!{S<:Union{MeanPhaseDiff{true}, PLV{true}, PPC{true}}, T<:Real}(t::AbstractJackknifeSurrogates{S},
                                                                                    out::JackknifeSurrogatesOutput,
                                                                                    work::Matrix{Complex{T}},
                                                                                    X::AbstractVecOrMat{Complex{T}},
                                                                                    Y::AbstractVecOrMat{Complex{T}})
    trueval = out.trueval
    surrogates = out.surrogates

    chkinput(trueval, X, Y)
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    Ac_mul_B!(work, X, Y)

    n = ntrials(X)
    finish_xy!(S, trueval, work, n)

    @inbounds for k = 1:size(Y, 2), j = 1:size(X, 2)
        x = work[j, k]
        for i = 1:size(surrogates, 1)
            v = x
            for idel = (i-1)*jnn(t)+1:i*jnn(t)
                v -= conj(X[idel, j])*Y[idel, k]
            end
            surrogates[i, j, k] = finish(S, v, n-jnn(t))
        end
    end

    out
end

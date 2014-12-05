#
# Coherency
#

immutable Coherency <: ComplexPairwiseStatistic; end

# Single input matrix
allocwork{T<:Real}(::Coherency, X::AbstractMatrix{Complex{T}}) = nothing
computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}}, work::Nothing,
                      X::AbstractMatrix{Complex{T}}) = 
    cov2coh!(out, A_mul_Ac!(out, X), ConjFun())

# Two input matrices
allocwork{T<:Real}(::Coherency, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    (Array(T, size(X, 1)), Array(T, size(Y, 1)))
function computestat!{T<:Real}(::Coherency, out::AbstractMatrix{Complex{T}},
                      work::(Vector{T}, Vector{T}),
                      X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    cov2coh!(out, X, Y, work[1], work[2], A_mul_Bc!(out, X, Y), ConjFun())
end

#
# Coherence (as the square root of the correlation matrix)
#

immutable Coherence <: RealPairwiseStatistic; end
allocwork{T<:Real}(::Coherence, X::AbstractMatrix{Complex{T}}) =
    Array(Complex{T}, size(X, 1), size(X, 1))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::Matrix{Complex{T}},
                      X::AbstractMatrix{Complex{T}}) =
    cov2coh!(out, A_mul_Ac!(work, X), Base.AbsFun())

allocwork{T<:Real}(::Coherence, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 1)), Array(T, size(X, 1)), Array(T, size(Y, 1)))
computestat!{T<:Real}(::Coherence, out::AbstractMatrix{T},
                      work::(Matrix{Complex{T}}, Vector{T}, Vector{T}),
                      X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) = 
    cov2coh!(out, X, Y, work[2], work[3], A_mul_Bc!(work[1], X, Y), Base.AbsFun())


# Jackknifing for Coherency/Coherence
allocwork{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                   X::AbstractMatrix{Complex{T}}) =
    (allocwork(t.transform, X), Array(T, size(X, 1)))
surrogateval(::Jackknife{Coherence}, v1, v2, m1, m2) = abs(v)*m1*m2
surrogateval(::Jackknife{Coherency}, v1, v2, m1, m2) = conj(v)*(m1*m2)
function computestat!{T<:Real}(t::Union(Jackknife{Coherency}, Jackknife{Coherence}),
                               out::JackknifeOutput,
                               work::(Union(Matrix{Complex{T}}, Nothing), Vector{T}),
                               X::AbstractMatrix{Complex{T}})
    stat = t.transform
    trueval = out.trueval
    surrogates = out.surrogates
    XXc::Matrix{Complex{T}} = isa(t, Jackknife{Coherence}) ? work[1] : trueval
    dg = work[2]

    A_mul_Ac!(XXc, X)

    # Surrogates
    @inbounds for k = 1:size(X, 2)
        for i = 1:size(X, 1)
            dg[i] = 1/sqrt(real(XXc[i, i]) - abs2(X[i, k]))
        end

        for j = 1:size(X, 1)
            x = dg[j]
            z = conj(X[j, k])
            for i = 1:j-1
                v = XXc[i, j] - X[i, k]*z
                if isa(t, Jackknife{Coherence})
                    surrogates[i, j, k] = abs(v)*real(dg[i])*x
                else
                    surrogates[i, j, k] = conj(v)*(real(dg[i])*x)
                end
            end
            surrogates[j, j, k] = one(T)
        end
    end

    # Finish true value
    cov2coh!(trueval, XXc, isa(t, Jackknife{Coherence}) ? Base.AbsFun() : ConjFun())

    out
end

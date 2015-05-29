#
# Uniform scores test statistic
#
# Mardia, K. V. (1972). A Multi-Sample Uniform Scores Test on a Circle
# and Its Parametric Competitor. Journal of the Royal Statistical
# Society. Series B (Methodological), 34(1), 102–113.
#
using StatsBase
immutable UniformScores <: PairwiseStatistic
    groups::Vector{Int32}
    ningroup::Vector{Int32}

    function UniformScores{T<:Integer}(groups::Vector{T})
        ext = extrema(groups)
        ext[1] == 1 || throw(ArgumentError("group indices must start at 1"))
        ningroup = zeros(Int32, ext[2])
        for i = 1:length(groups)
            ningroup[groups[i]] += 1
        end
        new(groups, ningroup)
    end
end
Base.eltype{T<:Real}(::UniformScores, X::AbstractArray{Complex{T}}) = T

allocwork{T<:Real}(t::UniformScores, X::AbstractVecOrMat{Complex{T}}) =
    (zeros(T, size(X)), zeros(T, ntrials(X)), zeros(Int, ntrials(X)), [cis(2π*i/size(X, 1)) for i = 1:size(X, 1)], zeros(Complex{T}, length(t.ningroup)))

# Single input matrix
using Base.Sort: DEFAULT_UNSTABLE, Perm, Forward
function computestat!{T<:Real}(t::UniformScores, out::AbstractMatrix{T},
                               work::Tuple{Matrix{T}, Vector{T}, Vector{Int}, Vector{Complex{T}}, Vector{Complex{T}}}, X::AbstractVecOrMat{Complex{T}})
    chkinput(out, X)
    angles, anglediff, ranks, circranks, d = work
    (size(X, 1) == size(angles, 1) == length(anglediff) == length(ranks) == length(circranks) == length(t.groups) &&
        size(X, 2) == size(angles, 2)) && length(d) == length(t.ningroup) || throw(ArgumentError("invalid work object"))
    for i = 1:length(X)
        @inbounds angles[i] = angle(X[i])
    end

    for k = 1:size(X, 2), j = 1:k-1
        @inbounds @simd for i = 1:size(X, 1)
            anglediff[i] = mod2pi(angles[i, j] - angles[i, k])
            ranks[i] = i
        end
        # sortperm!, but don't allocate memory
        sort!(ranks, DEFAULT_UNSTABLE, Perm(Forward, anglediff))
        fill!(d, zero(T))
        @simd for i = 1:size(X, 1)
            @inbounds d[t.groups[ranks[i]]] += circranks[i]
        end
        Wr = zero(T)
        @simd for i = 1:length(d)
            @inbounds Wr += abs2(d[i])/t.ningroup[i]
        end
        out[j, k] = 2*Wr
    end
    out
end

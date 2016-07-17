export computestat, computestat!, computestat_parallel, computestat_parallel!,
       PowerSpectrum, CrossSpectrum, Correlation, Coherency,
       Coherence, MeanPhaseDiff, PLV, PPC, PLI, PLI2Unbiased, WPLI,
       WPLI2Debiased, JammalamadakaR, JuppMardiaR,
       HurtadoModulationIndex, UniformScores, JackknifeSurrogates,
       MultiJackknifeSurrogates, jackknife_bias, jackknife_var,
       Jackknife, JackknifeCorrelation, Bootstrap, Permutation, GroupMean

#
# Utilities
#

typealias AbstractVecOrMat{T} Union{AbstractVector{T}, AbstractMatrix{T}}
typealias StridedVecOrMat{T} Union{StridedVector{T}, StridedMatrix{T}}

ntrials(X::AbstractVecOrMat) = size(X, 1)
nchannels(X::AbstractVecOrMat) = size(X, 2)

# Upper triangular symmetric/hermitian multiply, ignoring lower triangular part
Ac_mul_A!{T<:Complex}(out::StridedMatrix{T}, X::StridedVecOrMat{T}) =
    BLAS.herk!('U', 'C', real(one(T)), X, real(zero(T)), out)
At_mul_A!{T}(out::StridedMatrix{T}, X::StridedVecOrMat{T}) =
    BLAS.syrk!('U', 'C', real(one(T)), X, real(zero(T)), out)
Ac_mul_A!{T}(out::StridedMatrix{T}, X::StridedVecOrMat{T}) = At_mul_A!(out, X)

# Normalize to unit circle
function unitnormalize!{T<:Complex}(out::AbstractArray{T}, X::AbstractArray{T})
    length(out) == length(X) || error(DimensionMismatch("work size mismatch"))
    for i = 1:length(X)
        @inbounds out[i] = X[i]/abs(X[i])
    end
    out
end

# Convert uncentered covariance to coherency or coherence
function cov2coh!{T<:Real}(out::Union{AbstractMatrix{Complex{T}}, AbstractMatrix{T}},
                           XXc::Union{AbstractMatrix{Complex{T}}, AbstractMatrix{T}}=out,
                           f=Base.IdFun())
    size(XXc, 2) == size(XXc, 1) || error(ArgumentError("XXc must be square"))
    size(out, 1) == size(out, 2) == size(XXc, 2) ||
        error(DimensionMismatch("output size mismatch"))

    for j = 1:size(XXc, 2)
        out[j, j] = x = 1/sqrt(real(XXc[j, j]))
        for i = 1:j-1
            @inbounds out[i, j] = f(XXc[i, j])*(real(out[i, i])*x)
        end
    end
    for j = 1:size(XXc, 2)
        @inbounds out[j, j] = one(T)
    end
    out
end
function cov2coh!{T<:Real}(out::Union{AbstractMatrix{Complex{T}}, AbstractMatrix{T}},
                           X::AbstractVecOrMat, Y::AbstractVecOrMat,
                           Xtmp::AbstractVecOrMat{T}, Ytmp::AbstractVecOrMat{T},
                           XYc::Union{AbstractMatrix{Complex{T}}, AbstractMatrix{T}}=out,
                           f=Base.IdFun())
    (size(XYc, 1) == length(Xtmp) && size(XYc, 2) == length(Ytmp)) ||
        error(DimensionMismatch("work size mismatch"))
    (size(out, 1) == size(XYc, 1) && size(out, 2) == size(XYc, 2)) ||
        error(DimensionMismatch("output size mismatch"))

    sumabs2!(Xtmp, X)
    sumabs2!(Ytmp, Y)
    for i = 1:length(Xtmp)
        Xtmp[i] = 1/sqrt(Xtmp[i])
    end
    for i = 1:length(Ytmp)
        Ytmp[i] = 1/sqrt(Ytmp[i])
    end

    for j = 1:size(XYc, 2), i = 1:size(XYc, 1)
        @inbounds out[i, j] = f(XYc[i, j])*(Xtmp[i]*Ytmp[j])
    end
    out
end

# Work array for X or Y in two-argument form
cov2coh_work{T}(X::AbstractVector{Complex{T}}) = Array(T, 1)
cov2coh_work{T}(X::AbstractMatrix{Complex{T}}) = Array(T, 1, nchannels(X))
cov2coh_work{T}(X::AbstractVector{T}) = Array(T, 1)
cov2coh_work{T}(X::AbstractMatrix{T}) = Array(T, 1, nchannels(X))

# Fastest possible unsafe submatrix
immutable ConjFun <: Base.Func{1} end
call(::ConjFun, x) = conj(x)

# Check that X and Y have the same number of trials
chkXY(X, Y) = size(X, 1) == size(Y, 1) ||
    error(DimensionMismatch("X and Y must have same number of trials"))
chkout(out, X) = size(out, 1) == size(out, 2) == size(X, 2) ||
    error(DimensionMismatch("output size mismatch"))
chkout(out, X, Y) = (size(out, 1) == size(X, 2) && size(out, 2) == size(Y, 2)) ||
    error(DimensionMismatch("output size mismatch"))
chkinput(out, X) = chkout(out, X)
chkinput(out, X, Y) = (chkXY(X, Y); chkout(out, X, Y))

#
# Types and basics
#

abstract Statistic

abstract PairwiseStatistic <: Statistic
abstract NormalizedPairwiseStatistic{Normalized} <: PairwiseStatistic

# General definitions of allocoutput
Base.eltype{T,S}(s::PairwiseStatistic, X::AbstractArray{T}, Y::AbstractArray{S}) =
    promote_type(eltype(s, X), eltype(s, Y))
allocoutput{T}(t::PairwiseStatistic, X::AbstractVecOrMat{T}) =
    zeros(eltype(t, X), size(X, 2), size(X, 2))
allocoutput{T}(t::PairwiseStatistic, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    zeros(eltype(t, X, Y), size(X, 2), size(Y, 2))
outputdims{R<:PairwiseStatistic}(t::Type{R}) = 2

#
# Diagonal values
#
diagval{T<:Statistic}(::Type{T}) = nothing

#
# Handling of n-d arrays
#
# X is channels x trials x ...
# output is channels x channels x ...
computestat(t::Statistic, X::AbstractArray) =
    computestat!(t, allocoutput(t, X), allocwork(t, X), X)
computestat(t::Statistic, X::AbstractArray, Y::AbstractArray) =
    computestat!(t, allocoutput(t, X, Y), allocwork(t, X, Y), X, Y)
computestat_parallel(t::Statistic, X::AbstractArray) =
    computestat_parallel!(t, allocoutput(t, X), allocwork(t, X), X)
computestat_parallel(t::Statistic, X::AbstractArray, Y::AbstractArray) =
    computestat_parallel!(t, allocoutput(t, X, Y), allocwork(t, X, Y), X, Y)

allocwork{T}(t::Statistic, X::AbstractVecOrMat{T}) =
    error("allocwork($(typeof(t)), X) not defined")
allocwork{T}(t::Statistic, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    error("allocwork($(typeof(t)), X, Y) not defined")
allocwork{T}(t::Statistic, X::AbstractArray{T}) = allocwork(t, unsafe_view(X, :, :, 1))
function allocwork(t::Statistic, X::AbstractArray, Y::AbstractArray)
    allocwork(t, unsafe_view(X, :, :, 1), unsafe_view(Y, :, :, 1))
end

@generated function allocoutput{T,N}(t::Statistic, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(zeros(eltype(t, X, Y), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:N]...)))
end

computestat!{T}(t::Statistic, out::AbstractArray, work, X::AbstractVecOrMat{T}) =
    error("computestat! not defined for $(typeof(t)) with single input matrix, or incorrect work $(typeof(work)) or output $(typeof(out)) given")
computestat!{T}(t::Statistic, out::AbstractArray, work, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    error("computestat! not defined for $(typeof(t)) with two input matrices, or incorrect work $(typeof(work)) or output $(typeof(out)) given")

function checknd(t::Statistic, out::AbstractArray, X::AbstractArray)
    isempty(X) && throw(ArgumentError("X is empty"))
    trail = Base.trailingsize(X, 3)
    Base.trailingsize(out, outputdims(typeof(t))+1) == trail || throw(DimensionMismatch("output size mismatch"))
    trail
end

function checknd(t::Statistic, out::AbstractArray, X::AbstractArray, Y::AbstractArray)
    (isempty(X) || isempty(Y)) && throw(ArgumentError("X or Y is empty"))
    trail = Base.trailingsize(X, 3)
    Base.trailingsize(Y, 3) == trail || throw(DimensionMismatch("X and Y must have same trailing dimensions"))
    Base.trailingsize(out, outputdims(typeof(t))+1) == trail || throw(DimensionMismatch("output size mismatch"))
    trail
end

# Generate an unsafe_view of the output for the ith independent input
@generated function output_view(t::Statistic, out::AbstractArray, i::Union{Int, UnitRange{Int}})
    :(unsafe_view(out, $(fill(:(:), outputdims(t))...), i))
end

# nd computestat!
function computestat!{T,V}(t::Statistic, out::AbstractArray, work::V, X::AbstractArray{T})
    for i = 1:checknd(t, out, X)
        computestat!(t, output_view(t, out, i), work, unsafe_view(X, :, :, i))
    end
    out
end

function computestat!{T,V}(t::Statistic, out::AbstractArray, work::V, X::AbstractArray{T}, Y::AbstractArray{T})
    for i = 1:checknd(t, out, X, Y)
        computestat!(t, output_view(t, out, i), work, unsafe_view(X, :, :, i), unsafe_view(Y, :, :, i))
    end
    out
end

# parallel nd computestat!
function chunkranges(x::Int, n::Int)
    parts = Array(UnitRange{Int}, n)
    chunklen = div(x, n)
    curind = 1
    for i = 1:n
        chunklen = div(x - curind + 1, n-i+1)
        parts[i] = curind:curind+chunklen-1
        curind = curind+chunklen
    end
    parts
end

function computestat_parallel_complete!(t, out, refs, ranges)
    for i = 1:length(refs)
        res = fetch(refs[i])
        isa(res, Exception) && rethrow(res)
        copy!(output_view(t, out, ranges[i]), res)
    end
    out
end

function computestat_parallel!(t::Statistic, out, work, X::AbstractArray)
    np = max(nprocs()-1, 1)
    p = np == 1 ? (1:1) : 2:np+1
    ranges = chunkranges(checknd(t, out, X), np)
    n = length(ranges)

    refs = [remotecall(p[i], computestat, t, X[:, :, ranges[i]]) for i = 1:length(ranges)]
    computestat_parallel_complete!(t, out, refs, ranges)
end

function computestat_parallel!{T}(t::Statistic, out, work, X::AbstractArray{T}, Y::AbstractArray{T})
    np = max(nprocs()-1, 1)
    p = np == 1 ? (1:1) : 2:np+1
    ranges = chunkranges(checknd(t, out, X, Y), np)
    n = length(ranges)

    refs = [remotecall(p[i], computestat, t, X[:, :, ranges[i]], Y[:, :, ranges[i]]) for i = 1:length(ranges)]
    computestat_parallel_complete!(t, out, refs, ranges)
end

#
# Normalized statistics
#

@generated function normalized(t::NormalizedPairwiseStatistic{false})
    :($(t.name.name){true}())
end

allocwork{T<:Real}(t::NormalizedPairwiseStatistic{false}, X::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), allocwork(normalized(t), X))
allocwork{T<:Real}(t::NormalizedPairwiseStatistic{false}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), Array(Complex{T}, size(X, 1), size(X, 2)), allocwork(normalized(t), X, Y))
computestat!{T<:Real}(t::NormalizedPairwiseStatistic{false}, out::AbstractArray,
                      work::Tuple{Matrix{Complex{T}}, Any}, X::AbstractVecOrMat{Complex{T}}) =
    computestat!(normalized(t), out, work[2], unitnormalize!(work[1], X))
computestat!{T<:Real}(t::NormalizedPairwiseStatistic{false}, out::AbstractArray,
                      work::Tuple{Matrix{Complex{T}}, Matrix{Complex{T}}, Any},
                      X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    computestat!(normalized(t), out, work[3], unitnormalize!(work[1], X), unitnormalize!(work[2], Y))

#
# GroupMean
#

immutable GroupMean{R<:Statistic} <: Statistic
    transform::R
    indices::Vector{Int}
    offsets::Vector{Int}
    nchannels::Int
end
ngroups(x::GroupMean) = length(x.offsets)-1
outputdims{R<:GroupMean}(t::Type{R}) = 1

function GroupMean(t::Statistic, nchannels::Int, pairs::Vector{Vector{Tuple{Int,Int}}})
    indices = Int[]
    offsets = Int[1]
    for ipairgroup = 1:length(pairs)
        pairgroup = pairs[ipairgroup]
        for ipair in 1:size(pairgroup, 1)
            ch1, ch2 = pairgroup[ipair]
            (ch1 > nchannels || ch2 > nchannels || ch1 < 1 || ch2 < 1) && throw(ArgumentError("channel indices must be in range [1, nchannels"))
            if ch1 > ch2
                ch1, ch2 = ch2, ch1
            end
            push!(indices, (ch2-1)*nchannels+ch1)
        end
        push!(offsets, length(indices)+1)
    end
    GroupMean(t, indices, offsets, nchannels)
end

@generated function allocoutput{T,R<:PairwiseStatistic,N}(t::GroupMean{R}, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(zeros(eltype(t.transform, X, Y), ngroups(t), $([:(size(X, $i)) for i = 3:N]...)))
end

allocwork{T}(t::GroupMean, X::AbstractVecOrMat{T}) =
    (allocoutput(t.transform, X), allocwork(t.transform, X))
allocwork{T}(t::GroupMean, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (allocoutput(t.transform, X, Y), allocwork(t.transform, X, Y))

function grouppairs!{T}(t::GroupMean, out::AbstractVector{T}, chval::AbstractMatrix{T})
    length(out) == ngroups(t) || throw(ArgumentError("invalid output size"))
    @inbounds for ipairgroup = 1:ngroups(t)
        m = zero(T)
        @simd for ipair = t.offsets[ipairgroup]:t.offsets[ipairgroup+1]-1
            pairindex = t.indices[ipair]
            m += chval[pairindex]
        end
        out[ipairgroup] = m/(t.offsets[ipairgroup+1]-t.offsets[ipairgroup])
    end
    out
end

function computestat!{T}(t::GroupMean, out::AbstractArray, work::Tuple{Matrix, Any},
                         X::AbstractVecOrMat{T})
    chval, twork = work
    computestat!(t.transform, chval, twork, X)
    grouppairs!(t, out, chval)
    out
end
function computestat!{T}(t::GroupMean, out::AbstractArray, work::Tuple{Matrix, Any},
                         X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    chval, twork = work
    computestat!(t.transform, chval, twork, X, Y)
    grouppairs!(t, out, chval)
    out
end

#
# (Inefficient) generic jackknife surrogate computation
#

abstract AbstractJackknifeSurrogates{R<:Statistic} <: Statistic

# Jackknifing single samples
immutable JackknifeSurrogates{R<:Statistic} <: AbstractJackknifeSurrogates{R}
    transform::R
end
Base.similar(t::JackknifeSurrogates, stat::Statistic) = JackknifeSurrogates(stat)

# Jackknifing multiple samples (e.g. trials)
# This is NOT a delete-n jackknife; it creates only div(ntrials(X), n)
# samples
immutable MultiJackknifeSurrogates{R<:Statistic} <: AbstractJackknifeSurrogates{R}
    transform::R
    n::Int
end
Base.similar(t::MultiJackknifeSurrogates, stat::Statistic) = MultiJackknifeSurrogates(stat, t.n)

immutable JackknifeSurrogatesOutput{T<:StridedArray,S<:StridedArray}
    trueval::T
    surrogates::S
end

# Number of elements to be deleted at a time
jnn(::JackknifeSurrogates) = 1
jnn(t::MultiJackknifeSurrogates) = t.n

function copyrows!(out, outrow, X, rowinds)
    for j = 1:size(X, 2)
        i = 1
        for x in rowinds
            @inbounds out[outrow+i-1, j] = X[x, j]
            i += 1
        end
    end
    out
end

@generated function allocoutput{T,N}(t::AbstractJackknifeSurrogates, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(JackknifeSurrogatesOutput(zeros(eltype(t.transform, X, Y), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:N]...)),
                                zeros(eltype(t.transform, X, Y), div(size(X, 1), jnn(t)), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:N]...))))
end

function allocwork{T}(t::AbstractJackknifeSurrogates, X::AbstractVecOrMat{T})
    Xsurrogate = Array(T, size(X, 1)-jnn(t), size(X, 2))
    (Xsurrogate, allocoutput(t.transform, Xsurrogate), allocwork(t.transform, X), allocwork(t.transform, Xsurrogate))
end
function computestat!{T}(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, work,
                         X::AbstractVecOrMat{T})
    isa(work, Tuple{Matrix{T}, Any, Any, Any}) || throw(ArgumentError("invalid work object"))
    stat = t.transform
    ntrials, nch = size(X)
    njn = div(ntrials, jnn(t))
    Xsurrogate, outtmp, worktrue, worksurrogate = work
    trueval = out.trueval
    surrogates = out.surrogates

    (size(Xsurrogate, 1) == ntrials - jnn(t) && size(Xsurrogate, 2) == nch) || throw(ArgumentError("invalid work object"))
    ntrials % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    chkinput(trueval, X)

    fill!(surrogates, NaN)

    # True value
    computestat!(stat, trueval, worktrue, X)

    # First jackknife
    copyrows!(Xsurrogate, 1, X, jnn(t)+1:size(X, 1))
    computestat!(stat, outtmp, worksurrogate, Xsurrogate)
    copy!(unsafe_view(surrogates, 1, :, :), outtmp)

    # Subsequent jackknifes
    for i = 2:njn
        copyrows!(Xsurrogate, (i-2)*jnn(t)+1, X, (i-2)*jnn(t)+1:(i-1)*jnn(t))
        computestat!(stat, outtmp, worksurrogate, Xsurrogate)
        copy!(unsafe_view(surrogates, i, :, :), outtmp)
    end

    out
end

function allocwork{T}(t::AbstractJackknifeSurrogates, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    Xsurrogate = Array(T, size(X, 1)-jnn(t), size(X, 2))
    Ysurrogate = Array(T, size(Y, 1)-jnn(t), size(Y, 2))
    (Xsurrogate, Ysurrogate, allocoutput(t.transform, Xsurrogate, Ysurrogate),
     allocwork(t.transform, X, Y), allocwork(t.transform, Xsurrogate, Ysurrogate))
end
function computestat!{T}(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, work,
                         X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    isa(work, Tuple{Matrix{T}, Matrix{T}, Any, Any, Any}) || throw(ArgumentError("invalid work object"))
    stat = t.transform
    ntrials, nchX = size(X)
    njn = div(ntrials, jnn(t))
    nchY = size(Y, 2)
    Xsurrogate, Ysurrogate, outtmp, worktrue, worksurrogate = work
    trueval = out.trueval
    surrogates = out.surrogates

    (size(Xsurrogate, 1) == size(Ysurrogate, 1) == ntrials - jnn(t) &&
     size(Xsurrogate, 2) == nchX &&
     size(Ysurrogate, 2) == nchY) || throw(ArgumentError("invalid work object"))
    ntrials % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    ntrials > 0 || error("X is empty")
    chkinput(trueval, X, Y)

    fill!(surrogates, NaN)

    # True value
    computestat!(stat, trueval, worktrue, X, Y)

    # First jackknife
    copyrows!(Xsurrogate, 1, X, jnn(t)+1:size(X, 1))
    copyrows!(Ysurrogate, 1, Y, jnn(t)+1:size(X, 1))
    computestat!(stat, outtmp, worksurrogate, Xsurrogate, Ysurrogate)
    copy!(unsafe_view(surrogates, 1, :, :), outtmp)

    # Subsequent jackknifes
    for i = 2:njn
        copyrows!(Xsurrogate, (i-2)*jnn(t)+1, X, (i-2)*jnn(t)+1:(i-1)*jnn(t))
        copyrows!(Ysurrogate, (i-2)*jnn(t)+1, Y, (i-2)*jnn(t)+1:(i-1)*jnn(t))
        computestat!(stat, outtmp, worksurrogate, Xsurrogate, Ysurrogate)
        copy!(unsafe_view(surrogates, i, :, :), outtmp)
    end

    out
end

# Only unit normalize once
allocwork{R<:NormalizedPairwiseStatistic{false},T<:Real}(t::AbstractJackknifeSurrogates{R}, X::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), allocwork(similar(t, normalized(t.transform)), X))
allocwork{R<:NormalizedPairwiseStatistic{false},T<:Real}(t::AbstractJackknifeSurrogates{R}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), Array(Complex{T}, size(Y, 1), size(Y, 2)), allocwork(similar(t, normalized(t.transform)), X, Y))
computestat!{R<:NormalizedPairwiseStatistic{false},T<:Real}(t::AbstractJackknifeSurrogates{R}, out::JackknifeSurrogatesOutput,
                                                            work, X::AbstractVecOrMat{Complex{T}}) =
    computestat!(similar(t, normalized(t.transform)), out, work[2], unitnormalize!(work[1], X))
computestat!{R<:NormalizedPairwiseStatistic{false},T<:Real}(t::AbstractJackknifeSurrogates{R}, out::JackknifeSurrogatesOutput,
                                                            work, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    computestat!(similar(t, normalized(t.transform)), out, work[3], unitnormalize!(work[1], X), unitnormalize!(work[2], Y))

# Estimate of variance from jackknife surrogates
function _jackknife_var!{T}(out::AbstractArray{T}, surrogates::AbstractArray{T})
    n = size(surrogates, 1)
    for j = 1:Base.trailingsize(surrogates, 2)
        # Compute mean
        m = zero(eltype(surrogates))
        @simd for i = 1:n
            @inbounds m += surrogates[i, j]
        end
        m /= n

        # Compute variance
        v = zero(eltype(out))
        @simd for i = 1:n
            @inbounds v += abs2(surrogates[i, j] - m)
        end

        @inbounds out[j] = v*(n-1)/n
    end
    out
end
function jackknife_var!{T}(out::AbstractArray{T}, jn::JackknifeSurrogatesOutput)
    size(out) == size(jn.trueval) || throw(ArgumentError("output size mismatch"))
    _jackknife_var!(out, jn.surrogates)
    out
end
jackknife_var(jn::JackknifeSurrogatesOutput) = jackknife_var!(similar(jn.trueval), jn)

# Estimate of (first-order) bias from jackknife surrogates
function jackknife_bias!(out::Matrix, jn::JackknifeSurrogatesOutput)
    trueval = jn.trueval
    surrogates = jn.surrogates
    n = size(surrogates, 1)

    for k = 1:size(surrogates, 3), j = 1:size(surrogates, 2)
        v = zero(eltype(out))
        @simd for i = 1:n
            @inbounds v += surrogates[i, j, k]
        end
        @inbounds out[j, k] = (n - 1)*(v/n - trueval[j, k])
    end
    out
end
jackknife_bias(jn::JackknifeSurrogatesOutput) = jackknife_bias!(similar(jn.trueval), jn)

# Jackknifing of n-d arrays
function computestat!(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, work, X::AbstractArray)
    !isempty(X) || error(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        computestat!(t, JackknifeSurrogatesOutput(unsafe_view(out.trueval, :, :, i), unsafe_view(out.surrogates, :, :, :, i)),
                     work, unsafe_view(X, :, :, i))
    end
    out
end
function computestat!{T}(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, work, X::AbstractArray{T}, Y::AbstractArray{T})
    !isempty(X) || error(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        computestat!(t, JackknifeSurrogatesOutput(unsafe_view(out.trueval, :, :, i), unsafe_view(out.surrogates, :, :, :, i)),
                     work, unsafe_view(X, :, :, i), unsafe_view(Y, :, :, i))
    end
    out
end
checknd(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, X::AbstractArray) = checknd(t.transform, out.trueval, X)
checknd(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, X::AbstractArray, Y::AbstractArray) = checknd(t.transform, out.trueval, X, Y)
function computestat_parallel_complete!(t::AbstractJackknifeSurrogates, out::JackknifeSurrogatesOutput, refs, ranges)
    for i = 1:length(refs)
        res = fetch(refs[i])
        isa(res, Exception) && rethrow(res)
        copy!(unsafe_view(out.trueval, :, :, ranges[i]), res.trueval)
        copy!(unsafe_view(out.surrogates, :, :, :, ranges[i]), res.surrogates)
    end
    out
end

#
# Computation of just the jackknife trueval and variance
#

immutable Jackknife{R<:Statistic,S} <: Statistic
    transform::S
end
Jackknife(t::Statistic) =
    Jackknife{typeof(t),JackknifeSurrogates{typeof(t)}}(JackknifeSurrogates(t))
Jackknife(t::Statistic, n::Int) =
    Jackknife{typeof(t),MultiJackknifeSurrogates{typeof(t)}}(MultiJackknifeSurrogates(t, n))

immutable JackknifeOutput{T<:StridedArray}
    trueval::T
    var::T
end

@generated function allocoutput{T,N}(t::Jackknife, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(JackknifeOutput(zeros(eltype(t.transform.transform, X, Y), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:N]...)),
                      zeros(eltype(t.transform.transform, X, Y), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:N]...))))
end

allocwork{T}(t::Jackknife, X::AbstractVecOrMat{T}) =
    (allocwork(t.transform, X), zeros(eltype(t.transform.transform, X), div(size(X, 1), jnn(t.transform)), size(X, 2), size(X, 2)))
allocwork{T}(t::Jackknife, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (allocwork(t.transform, X, Y), zeros(eltype(t.transform.transform, X, Y), div(size(X, 1), jnn(t.transform)), size(X, 2), size(Y, 2)))

function computestat!{T,S}(t::Jackknife, out::JackknifeOutput, work::Tuple{Any, Array{S,3}},
                                 X::AbstractVecOrMat{T})
    surrogates = JackknifeSurrogatesOutput(out.trueval, work[2])
    computestat!(t.transform, surrogates, work[1], X)
    jackknife_var!(out.var, surrogates)
    out
end
function computestat!{T,S}(t::Jackknife, out::JackknifeOutput, work::Tuple{Any, Array{S,3}},
                                 X::AbstractVecOrMat{T},
                                 Y::AbstractVecOrMat{T})
    surrogates = JackknifeSurrogatesOutput(out.trueval, work[2])
    computestat!(t.transform, surrogates, work[1], X, Y)
    jackknife_var!(out.var, surrogates)
    out
end

# Jackknifing of n-d arrays
function computestat!{S,T}(t::Jackknife, out::JackknifeOutput, work::S, X::AbstractArray{T})
    !isempty(X) || error(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        surrogates = JackknifeSurrogatesOutput(unsafe_view(out.trueval, :, :, i), work[2])
        computestat!(t.transform, surrogates, work[1], unsafe_view(X, :, :, i))
        jackknife_var!(unsafe_view(out.var, :, :, i), surrogates)
    end
    out
end
function computestat!{S,T}(t::Jackknife, out::JackknifeOutput, work::S, X::AbstractArray{T}, Y::AbstractArray{T})
    !isempty(X) || error(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        surrogates = JackknifeSurrogatesOutput(unsafe_view(out.trueval, :, :, i), work[2])
        computestat!(t.transform, surrogates, work[1], unsafe_view(X, :, :, i), unsafe_view(Y, :, :, i))
        jackknife_var!(unsafe_view(out.var, :, :, i), surrogates)
    end
    out
end
checknd(t::Jackknife, out::JackknifeOutput, X::AbstractArray) = checknd(t.transform.transform, out.trueval, X)
checknd(t::Jackknife, out::JackknifeOutput, X::AbstractArray, Y::AbstractArray) = checknd(t.transform.transform, out.trueval, X, Y)
function computestat_parallel_complete!(t::Jackknife, out::JackknifeOutput, refs, ranges)
    for i = 1:length(refs)
        res = fetch(refs[i])
        isa(res, Exception) && rethrow(res)
        copy!(unsafe_view(out.trueval, :, :, ranges[i]), res.trueval)
        copy!(unsafe_view(out.var, :, :, ranges[i]), res.trueval)
    end
    out
end

#
# Correlation of jackknife surrogates with another statistic
#

immutable JackknifeCorrelation{R<:Statistic,S,T<:AbstractFloat,F} <: PairwiseStatistic
    transform::S
    y::Vector{T}
    yssq::T
    f::F

    function JackknifeCorrelation(transform, y, f)
        μy = mean(y)
        yssq = zero(eltype(y))
        centeredy = zeros(eltype(y), length(y))
        @inbounds @simd for i = 1:length(y)
            centeredy[i] = y[i] - μy
            yssq += abs2(y[i] - μy)
        end
        new(transform, centeredy, yssq, f)
    end
end
JackknifeCorrelation(t::PairwiseStatistic, y::AbstractVector, f=Base.IdFun()) =
    JackknifeCorrelation{typeof(t),JackknifeSurrogates{typeof(t)},eltype(y),typeof(f)}(JackknifeSurrogates(t), float(y), f)
JackknifeCorrelation(t::PairwiseStatistic, n::Int, y::AbstractVector, f=Base.IdFun()) =
    JackknifeCorrelation{typeof(t),MultiJackknifeSurrogates{typeof(t)},eltype(y),typeof(f)}(MultiJackknifeSurrogates(t, n), float(y), f)

Base.eltype{T<:Real}(::JackknifeCorrelation, X::AbstractArray{Complex{T}}) = T
Base.eltype{T<:Real}(::JackknifeCorrelation, X::AbstractArray{T}) = T

function allocwork(t::JackknifeCorrelation, X::AbstractVecOrMat)
    outtype = eltype(t.transform.transform, X)
    outtype <: Real || throw(ArgumentError("JackknifeCorrelation defined only for statistics with real outputs"))
    (zeros(outtype, div(size(X, 1), jnn(t.transform)), size(X, 2), size(X, 2)), allocwork(t.transform, X))
end
function allocwork{T}(t::JackknifeCorrelation, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    outtype = eltype(t.transform.transform, X)
    outtype <: Real || throw(ArgumentError("JackknifeCorrelation defined only for statistics with real outputs"))
    (zeros(outtype, div(size(X, 1), jnn(t.transform)), size(X, 2), size(Y, 2)), allocwork(t.transform, X, Y))
end

function jackknife_cor!{T<:Real}(t::JackknifeCorrelation, out::AbstractVecOrMat{T}, surrogates::AbstractArray{T,3})
    y = t.y
    @inbounds for itrail = 1:Base.trailingsize(surrogates, 2)
        μx = zero(T)
        @simd for isurrogate = 1:size(surrogates, 1)
            surrogates[isurrogate, itrail] = t.f(surrogates[isurrogate, itrail])
            μx += surrogates[isurrogate, itrail]
        end
        μx /= size(surrogates, 1)
        cov = zero(T)
        xssq = zero(T)
        @simd for isurrogate = 1:size(surrogates, 1)
            centeredx = surrogates[isurrogate, itrail] - μx
            cov += centeredx*y[isurrogate]
            xssq += abs2(centeredx)
        end
        out[itrail] = -cov/sqrt(xssq*t.yssq)
    end
    out
end

function computestat!{T<:Real}(t::JackknifeCorrelation, out::AbstractMatrix{T}, work::Tuple{Array{T,3},Any},
                               X::AbstractVecOrMat)
    length(t.y) == div(size(X, 1), jnn(t.transform)) || throw(DimensionMismatch("length of y does not match number of surrogates"))
    surrogates = JackknifeSurrogatesOutput(out, work[1])
    computestat!(t.transform, surrogates, work[2], X)
    jackknife_cor!(t, out, work[1])
end
function computestat!{T<:Real,S}(t::JackknifeCorrelation, out::AbstractMatrix{T}, work::Tuple{Array{T,3},Any},
                                 X::AbstractVecOrMat{S}, Y::AbstractVecOrMat{S})
    length(t.y) == div(size(X, 1), jnn(t.transform)) || throw(DimensionMismatch("length of y does not match number of surrogates"))
    surrogates = JackknifeSurrogatesOutput(out, work[1])
    computestat!(t.transform, surrogates, work[2], X, Y)
    jackknife_cor!(t, out, work[1])
end

#
# Jackknife{GroupMean}
#

@generated function allocoutput{T,R<:Statistic,N}(t::Jackknife{GroupMean{R}}, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(JackknifeOutput(zeros(eltype(t.transform.transform.transform, X, Y), ngroups(t.transform.transform), $([:(size(X, $i)) for i = 3:N]...)),
                      zeros(eltype(t.transform.transform.transform, X, Y), ngroups(t.transform.transform), $([:(size(X, $i)) for i = 3:N]...))))
end

allocwork{R<:Statistic}(t::Jackknife{GroupMean{R}}, X::AbstractVecOrMat) =
    (allocwork(JackknifeSurrogates(t.transform.transform.transform), X),
     allocoutput(JackknifeSurrogates(t.transform.transform.transform), X),
     zeros(eltype(t.transform.transform.transform, X), size(X, 1), ngroups(t.transform.transform)))
allocwork{T,R<:Statistic}(t::Jackknife{GroupMean{R}}, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (allocwork(JackknifeSurrogates(t.transform.transform.transform), X, Y),
     allocoutput(JackknifeSurrogates(t.transform.transform.transform), X, Y),
     zeros(eltype(t.transform.transform.transform, X), size(X, 1), ngroups(t.transform.transform)))

function grouppairs!{T,R<:Statistic}(t::Jackknife{GroupMean{R}}, group_trueval::AbstractArray{T},
                                     group_var::AbstractArray{T}, group_surrogates::AbstractMatrix{T},
                                     jn::JackknifeSurrogatesOutput)
    fill!(group_surrogates, zero(T))
    jnt = t.transform.transform

    # First compute means of groups and jackknife surrogates
    for ipairgroup = 1:ngroups(jnt)
        m = zero(T)
        for ipair = jnt.offsets[ipairgroup]:jnt.offsets[ipairgroup+1]-1
            pairindex = jnt.indices[ipair]
            m += jn.trueval[pairindex]
            @simd for isurrogate = 1:size(jn.surrogates, 1)
                group_surrogates[isurrogate, ipairgroup] += jn.surrogates[isurrogate, pairindex]
            end
        end
        group_trueval[ipairgroup] = m/(jnt.offsets[ipairgroup+1]-jnt.offsets[ipairgroup])
    end

    # Now compute jackknife variances
    _jackknife_var!(group_var, group_surrogates)

    # Adjust for taking the means
    for ipairgroup = 1:ngroups(jnt)
        group_var[ipairgroup] /= abs2(jnt.offsets[ipairgroup+1]-jnt.offsets[ipairgroup])
    end
    nothing
end

function computestat!{R<:Statistic,S}(t::Jackknife{GroupMean{R}}, out::JackknifeOutput,
                                      work::Tuple{Any, JackknifeSurrogatesOutput{Array{S,2}, Array{S,3}}, Array{S,2}},
                                      X::AbstractVecOrMat)
    computestat!(JackknifeSurrogates(t.transform.transform.transform), work[2], work[1], X)
    grouppairs!(t, out.trueval, out.var, work[3], work[2])
    out
end
function computestat!{R<:Statistic,T,S}(t::Jackknife{GroupMean{R}}, out::JackknifeOutput,
                                              work::Tuple{Any, JackknifeSurrogatesOutput{Array{S,2}, Array{S,3}}, Array{S,2}},
                                              X::AbstractVecOrMat{T},
                                              Y::AbstractVecOrMat{T})
    computestat!(JackknifeSurrogates(t.transform.transform.transform), work[2], work[1], X, Y)
    grouppairs!(t, out.trueval, out.var, work[3], work[2])
    out
end

# Group jackknifing of n-d arrays
function computestat!{R<:Statistic,S}(t::Jackknife{GroupMean{R}}, out::JackknifeOutput,
                                      work::Tuple{Any, JackknifeSurrogatesOutput{Array{S,2}, Array{S,3}}, Array{S,2}}, X::AbstractArray)
    !isempty(X) || throw(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        computestat!(JackknifeSurrogates(t.transform.transform.transform), work[2], work[1], unsafe_view(X, :, :, i))
        grouppairs!(t, unsafe_view(out.trueval, :, i), unsafe_view(out.var, :, i), work[3], work[2])
    end
    out
end
function computestat!{R<:Statistic,T,S}(t::Jackknife{GroupMean{R}}, out::JackknifeOutput,
                                        work::Tuple{Any, JackknifeSurrogatesOutput{Array{S,2}, Array{S,3}}, Array{S,2}}, X::AbstractArray{T}, Y::AbstractArray{T})
    !isempty(X) || throw(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        computestat!(JackknifeSurrogates(t.transform.transform.transform), work[2], work[1], unsafe_view(X, :, :, i), unsafe_view(Y, :, :, i))
        grouppairs!(t, unsafe_view(out.trueval, :, i), unsafe_view(out.var, :, i), work[3], work[2])
    end
    out
end

#
# Bootstrap and Permutation definitions and common functionality
#

immutable Bootstrap{R<:Statistic} <: Statistic
    transform::R
    indices::Matrix{Int32}    # Sample weights for each bootstrap. ntrials x nbootstraps
end
Bootstrap{T<:Integer}(t::Statistic, indices::Matrix{T}) = Bootstrap(t, convert(Matrix{Int32}, indices))
Bootstrap(transform::Statistic, ntrials::Int, nbootstraps::Int) =
    Bootstrap(transform, rand(Int32(1):Int32(ntrials), ntrials, nbootstraps))

immutable Permutation{S} <: Statistic
    transform::S
    indices::Matrix{Int32}
end
function Permutation(transform::Statistic, ntrials::Int, nperm::Int)
    indices = Array(Int32, ntrials, nperm)
    v = [1:ntrials;]
    for iperm = 1:nperm
        indices[:, iperm] = shuffle!(v)
    end
    Permutation(transform, indices)
end

outputdims{R<:Statistic}(t::Union{Type{Bootstrap{R}}, Type{Permutation{R}}}) = outputdims(R)+1
@generated function allocoutput{T,N}(t::Union{Bootstrap, Permutation}, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(zeros(eltype(t.transform, X, Y), size(X, 2), size(Y, 2), size(t.indices, 2), $([:(size(X, $i)) for i = 3:N]...)))
end
@generated function allocoutput{T,R<:Statistic,N}(t::Union{Bootstrap{GroupMean{R}}, Permutation{GroupMean{R}}}, X::AbstractArray{T,N}, Y::AbstractArray{T,N}=X)
    :(zeros(eltype(t.transform.transform, X, Y), ngroups(t.transform), size(t.indices, 2), $([:(size(X, $i)) for i = 3:N]...)))
end

# Copy `indices` from X to out
function copyindices!(out, X, indices, iindices)
    @inbounds for j = 1:size(X, 2)
        @simd for i = 1:size(indices, 1)
            @inbounds out[i, j] = X[indices[i, iindices], j]
        end
    end
    out
end

#
# Bootstrapping
#

allocwork{T}(t::Bootstrap, X::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), allocwork(t.transform, X))
function bootstrap!{T,U}(t::Statistic, indices::Matrix{Int32}, out::AbstractArray,
                         Xbootstrap::Matrix{T}, work::U, X::AbstractVecOrMat{T})
    size(X, 1) == size(indices, 1) || throw(ArgumentError("number of bootstrap trials does not match number of trials"))
    (size(X, 1) == size(Xbootstrap, 1) &&
     size(X, 2) == size(Xbootstrap, 2)) || throw(ArgumentError("invalid work object"))

    for ibootstrap = 1:size(indices, 2)
        copyindices!(Xbootstrap, X, indices, ibootstrap)
        computestat!(t, output_view(t, out, ibootstrap), work, Xbootstrap)
    end
    out
end
computestat!{R<:Statistic,T}(t::Bootstrap{R}, out::AbstractArray,
                             work::Tuple{Matrix{T}, Any},
                             X::AbstractVecOrMat{T}) =
    bootstrap!(t.transform, t.indices, out, work[1], work[2], X)

allocwork{T}(t::Bootstrap, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), Array(T, size(Y, 1), size(Y, 2)), allocwork(t.transform, X, Y))
function bootstrap!{T,U}(t::Statistic, indices::Matrix{Int32}, out::AbstractArray,
                         Xbootstrap::Matrix{T}, Ybootstrap::Matrix{T}, work::U,
                         X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    size(X, 1) == size(indices, 1) || throw(ArgumentError("number of bootstrap trials does not match number of trials"))
    (size(X, 1) == size(Xbootstrap, 1) &&
     size(Y, 1) == size(Ybootstrap, 1) &&
     size(X, 2) == size(Xbootstrap, 2) &&
     size(Y, 2) == size(Ybootstrap, 2)) || throw(ArgumentError("invalid work object"))

    for ibootstrap = 1:size(indices, 2)
        copyindices!(Xbootstrap, X, indices, ibootstrap)
        copyindices!(Ybootstrap, Y, indices, ibootstrap)
        computestat!(t, output_view(t, out, ibootstrap), work, Xbootstrap, Ybootstrap)
    end
    out
end
computestat!{R<:Statistic,T}(t::Bootstrap{R}, out::AbstractArray,
                             work::Tuple{Matrix{T}, Matrix{T}, Any},
                             X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    bootstrap!(t.transform, t.indices, out, work[1], work[2], work[3], X, Y)

# Only unit normalize once
computestat!{R<:NormalizedPairwiseStatistic{false},T}(t::Bootstrap{R}, out::AbstractArray,
                                                      work::Tuple{Matrix{T}, Any},
                                                      X::AbstractVecOrMat{T}) =
    bootstrap!(normalized(t.transform), t.indices, out, work[1], work[2][2], unitnormalize!(work[2][1], X))
computestat!{R<:NormalizedPairwiseStatistic{false},T}(t::Bootstrap{R}, out::AbstractArray,
                                                      work::Tuple{Matrix{T}, Matrix{T}, Any},
                                                      X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    bootstrap!(normalized(t.transform), t.indices, out, work[1], work[2], work[3][3],
               unitnormalize!(work[3][1], X), unitnormalize!(work[3][2], Y))

#
# Permutation
#

allocwork{T}(t::Permutation, X::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), allocwork(t.transform, X, X))
# Separate allocwork method here to avoid allocating a second array for normalizing X
allocwork{T,R<:NormalizedPairwiseStatistic{false}}(t::Permutation{R}, X::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), Array(T, size(X, 1), size(X, 2)), allocwork(normalized(t.transform), X, X))
function permutation!{T,V}(t::Statistic, indices::Matrix{Int32}, out::AbstractArray, perm::Array{T,2}, work::V,
                           X::AbstractVecOrMat{T})
    size(X, 1) == size(indices, 1) || throw(ArgumentError("number of permutation trials does not match number of trials"))
    (size(X, 1) == size(perm, 1) &&
     size(X, 2) == size(perm, 2)) || throw(ArgumentError("invalid work object"))

    for iperm = 1:size(indices, 2)
        copyindices!(perm, X, indices, iperm)
        computestat!(t, output_view(t, out, iperm), work, perm, X)
    end
    out
end
computestat!{R<:Statistic,T}(t::Permutation{R}, out::AbstractArray, work::Tuple{Matrix{T}, Any},
                             X::AbstractVecOrMat{T}) =
    permutation!(t.transform, t.indices, out, work[1], work[2], X)

allocwork{T}(t::Permutation, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    (Array(T, size(X, 1), size(X, 2)), allocwork(t.transform, X, Y))
function permutation!{T,V}(t::Statistic, indices::Matrix{Int32}, out::AbstractArray, perm::Array{T,2}, work::V,
                           X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T})
    size(X, 1) == size(indices, 1) || throw(ArgumentError("number of permutation trials does not match number of trials"))
    (size(X, 1) == size(perm, 1) &&
     size(X, 2) == size(perm, 2)) || throw(ArgumentError("invalid work object"))

    for iperm = 1:size(indices, 2)
        copyindices!(perm, X, indices, iperm)
        computestat!(t, output_view(t, out, iperm), work, perm, Y)
    end
    out
end
computestat!{R<:Statistic,T}(t::Permutation{R}, out::AbstractArray, work::Tuple{Matrix{T}, Any},
                             X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    permutation!(t.transform, t.indices, out, work[1], work[2], X, Y)

# Only unit normalize once
computestat!{R<:NormalizedPairwiseStatistic{false},T}(t::Permutation{R}, out::AbstractArray,
                                                      work::Tuple{Matrix{T}, Matrix{T}, Any},
                                                      X::AbstractVecOrMat{T}) =
    permutation!(normalized(t.transform), t.indices, out, work[2], work[3], unitnormalize!(work[1], X))
computestat!{R<:NormalizedPairwiseStatistic{false},T}(t::Permutation{R}, out::AbstractArray,
                                                      work::Tuple{Matrix{T}, Any},
                                                      X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    permutation!(normalized(t.transform), t.indices, out, work[1], work[2][3],
               unitnormalize!(work[2][1], X), unitnormalize!(work[2][2], Y))

#
# Efficient permutation for JackknifeCorrelation
#

function perm_jackknife_cor!{T<:Real}(t::JackknifeCorrelation, out::AbstractArray{T}, surrogates::AbstractArray{T,3}, permy::Matrix)
    y = t.y
    trailsize = Base.trailingsize(surrogates, 2)
    @inbounds for itrail = 1:trailsize
        μx = zero(T)
        @simd for isurrogate = 1:size(surrogates, 1)
            surrogates[isurrogate, itrail] = t.f(surrogates[isurrogate, itrail])
            μx += surrogates[isurrogate, itrail]
        end
        μx /= size(surrogates, 1)

        xssq = zero(T)
        @simd for isurrogate = 1:size(surrogates, 1)
            surrogates[isurrogate, itrail] -= μx
            xssq += abs2(surrogates[isurrogate, itrail])
        end

        for iperm = 1:size(permy, 2)
            cov = zero(T)
            @simd for isurrogate = 1:size(surrogates, 1)
                cov += surrogates[isurrogate, itrail]*permy[isurrogate, iperm]
            end
            out[(iperm-1)*trailsize+itrail] = -cov/sqrt(xssq*t.yssq)
        end
    end
    out
end


function invpermy(y, indices)
    size(indices, 1) == length(y) || throw(DimensionMismatch("length of indices does not match length of y"))
    permy = zeros(eltype(y), size(indices))
    for j = 1:size(indices, 2)
        @simd for i = 1:size(indices, 1)
            @inbounds permy[indices[i, j], j] = y[i]
        end
    end
    permy
end

immutable JackknifeCorrelationPermutationWork{T<:Real,S}
    yvals::Matrix{T}
    twork::S
end

allocwork{R<:JackknifeCorrelation}(t::Permutation{R}, X::AbstractVecOrMat) =
    JackknifeCorrelationPermutationWork(invpermy(t.transform.y, t.indices), allocwork(t.transform, X))
function computestat!{R<:JackknifeCorrelation}(t::Permutation{R}, out::AbstractArray, work::JackknifeCorrelationPermutationWork,
                                               X::AbstractVecOrMat)
    indices = t.indices
    yvals = work.yvals
    twork = work.twork
    njn = div(size(X, 1), jnn(t.transform.transform))
    njn == size(indices, 1) || throw(ArgumentError("number of permutation trials does not match number of trials"))
    njn == length(t.transform.y) || throw(DimensionMismatch("length of y does not match number of surrogates"))

    surrogates = JackknifeSurrogatesOutput(output_view(t.transform.transform.transform, out, 1), twork[1])
    computestat!(t.transform.transform, surrogates, twork[2], X)
    perm_jackknife_cor!(t.transform, out, twork[1], yvals)
end

allocwork{R<:JackknifeCorrelation,T}(t::Permutation{R}, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    JackknifeCorrelationPermutationWork(invpermy(t.transform.y, t.indices), allocwork(t.transform, X, Y))
function computestat!{R<:JackknifeCorrelation,S}(t::Permutation{R}, out::AbstractArray, work::JackknifeCorrelationPermutationWork,
                                                 X::AbstractVecOrMat{S}, Y::AbstractVecOrMat{S})
    indices = t.indices
    yvals = work.yvals
    twork = work.twork
    njn = div(size(X, 1), jnn(t.transform.transform))
    njn == size(indices, 1) || throw(ArgumentError("number of permutation trials does not match number of trials"))
    njn == length(t.transform.y) || throw(DimensionMismatch("length of y does not match number of surrogates"))

    surrogates = JackknifeSurrogatesOutput(output_view(t.transform.transform.transform, out, 1), twork[1])
    computestat!(t.transform.transform, surrogates, twork[2], X, Y)
    perm_jackknife_cor!(t.transform, out, twork[1], yvals)
end

#
# Transforms
#

include("transforms/spectrum.jl")
include("transforms/correlation.jl")
include("transforms/coherence.jl")
include("transforms/plv.jl")
include("transforms/pli.jl")
include("transforms/jammalamadaka.jl")
include("transforms/juppmardia.jl")
include("transforms/modulationindex.jl")
include("transforms/uniformscores.jl")

export computestat, computestat!, PowerSpectrum, CrossSpectrum, Coherency,
       Coherence, MeanPhaseDiff, PLV, PPC, PLI, PLI2Unbiased, WPLI,
       WPLI2Debiased, JammalamadakaR, JuppMardiaR,
       HurtadoModulationIndex, Jackknife, jackknife_bias, jackknife_var, Bootstrap,
       genweights

#
# Utilities
#

typealias AbstractVecOrMat{T} Union(AbstractVector{T}, AbstractMatrix{T})
typealias StridedVecOrMat{T} Union(StridedVector{T}, StridedMatrix{T})

ntrials(X::AbstractVecOrMat) = size(X, 1)
nchannels(X::AbstractVecOrMat) = size(X, 2)

# Upper triangular symmetric/hermitian multiply, ignoring lower triangular part
Ac_mul_A!{T}(out::StridedMatrix{T}, X::StridedVecOrMat{T}) =
    BLAS.herk!('U', 'C', one(T), X, zero(T), out)
At_mul_A!{T}(out::StridedMatrix{T}, X::StridedVecOrMat{T}) =
    BLAS.syrk!('U', 'C', one(T), X, zero(T), out)

# Normalize to unit circle
function unitnormalize!{T<:Complex}(out::AbstractArray{T}, X::AbstractArray{T})
    length(out) == length(X) || error(DimensionMismatch("work size mismatch"))
    for i = 1:length(X)
        @inbounds out[i] = X[i]/abs(X[i])
    end
    out
end

# Convert uncentered covariance to coherency or coherence
function cov2coh!{T<:Real}(out::Union(AbstractMatrix{Complex{T}}, AbstractMatrix{T}),
                           XXc::Union(AbstractMatrix{Complex{T}}, AbstractMatrix{T})=out,
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
function cov2coh!{T<:Real}(out::Union(AbstractMatrix{Complex{T}}, AbstractMatrix{T}),
                           X::AbstractVecOrMat, Y::AbstractVecOrMat,
                           Xtmp::AbstractVecOrMat{T}, Ytmp::AbstractVecOrMat{T},
                           XYc::Union(AbstractMatrix{Complex{T}}, AbstractMatrix{T})=out,
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

# Fastest possible unsafe submatrix
immutable UnsafeSubMatrix{T<:Number} <: DenseMatrix{T}
    ptr::Ptr{T}
    nrow::Int
    ncol::Int
end
Base.getindex(X::UnsafeSubMatrix, i::Int) = unsafe_load(X.ptr, i)
Base.getindex(X::UnsafeSubMatrix, i::Int, j::Int) = unsafe_load(X.ptr, X.nrow*(j-1)+i)
Base.setindex!{T}(X::UnsafeSubMatrix{T}, x, i::Int) = unsafe_store!(X.ptr, convert(T, x), i)
Base.setindex!{T}(X::UnsafeSubMatrix{T}, x, i::Int, j::Int) = unsafe_store!(X.ptr, convert(T, x), X.nrow*(j-1)+i)
Base.pointer(X::UnsafeSubMatrix) = X.ptr
Base.convert{T}(::Type{Ptr{T}}, X::UnsafeSubMatrix) = X.ptr
Base.size(X::UnsafeSubMatrix) = X.nrow, X.ncol
Base.similar(X::UnsafeSubMatrix, T, dims) = Array(T, dims)

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
abstract RealPairwiseStatistic <: PairwiseStatistic
abstract ComplexPairwiseStatistic <: PairwiseStatistic

# General definitions of allocoutput
Base.eltype{T<:Real}(::RealPairwiseStatistic, X::AbstractArray{Complex{T}}) = T
Base.eltype{T<:Real}(::ComplexPairwiseStatistic, X::AbstractArray{Complex{T}}) = Complex{T}
Base.eltype{T<:Real,S<:Real}(::RealPairwiseStatistic, X::AbstractArray{Complex{T}}, Y::AbstractArray{Complex{S}}) =
    promote_type(T, S)
Base.eltype{T<:Real,S<:Real}(::ComplexPairwiseStatistic, X::AbstractArray{Complex{T}}, Y::AbstractArray{Complex{S}}) =
    Complex{promote_type(T, S)}
allocoutput{T<:Real}(t::PairwiseStatistic, X::AbstractVecOrMat{Complex{T}}) =
    zeros(eltype(t, X), size(X, 2), size(X, 2))
allocoutput{T<:Real}(t::PairwiseStatistic, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    zeros(eltype(t, X, Y), size(X, 2), size(Y, 2))

#
# Handling of n-d arrays
#
# X is channels x trials x ...
# output is channels x channels x ...
computestat(t::Statistic, X::AbstractArray) =
    computestat!(t, allocoutput(t, X), allocwork(t, X), X)
computestat(t::Statistic, X::AbstractArray, Y::AbstractArray) =
    computestat!(t, allocoutput(t, X, Y), allocwork(t, X, Y), X, Y)

allocwork{T<:Real}(t::Statistic, X::AbstractVecOrMat{Complex{T}}) =
    error("allocwork($(typeof(t)), X) not defined")
allocwork{T<:Real}(t::Statistic, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    error("allocwork($(typeof(t)), X, Y) not defined")
allocwork{T<:Real}(t::Statistic, X::AbstractArray{Complex{T}}) = allocwork(t, UnsafeSubMatrix(pointer(X), size(X, 2), size(X, 2)))
function allocwork(t::Statistic, X::AbstractArray, Y::AbstractArray)
    allocwork(t, UnsafeSubMatrix(pointer(X), size(X, 2), size(X, 2)),
                 UnsafeSubMatrix(pointer(Y), size(Y, 2), size(Y, 2)))
end

for n = 3:6
    @eval begin
        allocoutput{T<:Complex}(t::Statistic, X::AbstractArray{T,$n}, Y::AbstractArray{T,$n}=X) =
            zeros(eltype(t, X), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:n]...))
    end
end

computestat!{T<:Complex}(t::Statistic, out::AbstractArray, work, X::AbstractVecOrMat{T}) =
    error("computestat! not defined for $(typeof(t)) with single input matrix, or incorrect work given")
computestat!{T<:Complex}(t::Statistic, out::AbstractArray, work, X::AbstractVecOrMat{T}, Y::AbstractVecOrMat{T}) =
    error("computestat! not defined for $(typeof(t)) with two input matrices, or incorrect work given")
function computestat!{T<:Complex}(t::Statistic, out::AbstractArray, work, X::AbstractArray{T})
    !isempty(X) || error(ArgumentError("X is empty"))
    lead = size(X, 1)*size(X, 2)
    leadout = size(X, 2)*size(X, 2)
    trail = Base.trailingsize(X, 3)
    Base.trailingsize(out, 3) == trail || error(DimensionMismatch("output size mismatch"))

    for i = 1:trail
        computestat!(t, UnsafeSubMatrix(pointer(out, leadout*(i-1)+1), size(out, 1), size(out, 2)),
                     work, UnsafeSubMatrix(pointer(X, lead*(i-1)+1), size(X, 1), size(X, 2)))
    end
    out
end

function computestat!{T<:Complex}(t::Statistic, out::AbstractArray, work, X::AbstractArray{T}, Y::AbstractArray{T})
    leadX = size(X, 1)*size(X, 2)
    leadY = size(Y, 1)*size(Y, 2)
    leadout = size(X, 2)*size(Y, 2)
    trail = Base.trailingsize(X, 3)
    Base.trailingsize(Y, 3) == trail || error(DimensionMismatch("X and Y must have same trailing dimensions"))
    Base.trailingsize(out, 3) == trail || error(DimensionMismatch("output size mismatch"))

    for i = 1:trail
        computestat!(t, UnsafeSubMatrix(pointer(out, leadout*(i-1)+1), size(out, 1), size(out, 2)),
                     work, UnsafeSubMatrix(pointer(X, leadX*(i-1)+1), size(X, 1), size(X, 2)),
                     UnsafeSubMatrix(pointer(Y, leadY*(i-1)+1), size(Y, 1), size(Y, 2)))
    end
    out
end

# If t.normalized is false, compute work.normalizedX (and maybe
# work.normalizedY) in place and return it. Otherwise return X.
normalized(t::PairwiseStatistic, work, X) =
    !t.normalized ? unitnormalize!(work.normalizedX, X) : X
function normalized(t::PairwiseStatistic, work, X, Y)
    if !t.normalized
        return (unitnormalize!(work.normalizedX, X), unitnormalize!(work.normalizedY, Y))
    else
        return (X, Y)
    end
end

#
# Inefficient generic jackknife surrogate computation
#

immutable Jackknife{R<:Statistic} <: Statistic
    transform::R
end
immutable JackknifeOutput{T<:StridedArray,S<:StridedArray}
    trueval::T
    surrogates::S
end

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

for n = 2:6
    @eval begin
        allocoutput{T<:Complex}(t::Jackknife, X::AbstractArray{T,$n}, Y::AbstractArray{T,$n}=X) =
            JackknifeOutput(zeros(eltype(t.transform, X, Y), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:n]...)),
                            zeros(eltype(t.transform, X, Y), size(X, 1), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:n]...)))
    end
end

function allocwork{T<:Real}(t::Jackknife, X::AbstractVecOrMat{Complex{T}})
    Xsurrogate = Array(Complex{T}, size(X, 1)-1, size(X, 2))
    (Xsurrogate, allocoutput(t.transform, Xsurrogate), allocwork(t.transform, X), allocwork(t.transform, Xsurrogate))
end
function computestat!{R<:Statistic,T<:Real,V}(t::Jackknife{R}, out::JackknifeOutput,
                                              work::(Matrix{Complex{T}}, Any, V, V),
                                              X::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    ntrials, nch = size(X)
    Xsurrogate, outtmp, worktrue, worksurrogate = work
    trueval = out.trueval
    surrogates = out.surrogates

    (size(Xsurrogate, 1) == ntrials - 1 && size(Xsurrogate, 2) == nch) || error("invalid work object")
    chkinput(trueval, X)

    fill!(surrogates, NaN)

    # True value
    computestat!(stat, trueval, worktrue, X)

    # First jackknife
    copyrows!(Xsurrogate, 1, X, 2:size(X, 1))
    computestat!(stat, outtmp, worksurrogate, Xsurrogate)
    surrogates[1, :, :] = outtmp

    # Subsequent jackknifes
    for i = 2:ntrials
        copyrows!(Xsurrogate, i-1, X, i-1)
        computestat!(stat, outtmp, worksurrogate, Xsurrogate)
        surrogates[i, :, :] = outtmp
    end

    out
end

function allocwork{T<:Real}(t::Jackknife, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    Xsurrogate = Array(Complex{T}, size(X, 1)-1, size(X, 2))
    Ysurrogate = Array(Complex{T}, size(Y, 1)-1, size(Y, 2))
    (Xsurrogate, Ysurrogate, allocoutput(t.transform, Xsurrogate, Ysurrogate),
     allocwork(t.transform, X, Y), allocwork(t.transform, Xsurrogate, Ysurrogate))
end
function computestat!{R<:Statistic,T<:Real,V}(t::Jackknife{R}, out::JackknifeOutput,
                                              work::(Matrix{Complex{T}}, Matrix{Complex{T}}, Any, V, V),
                                              X::AbstractVecOrMat{Complex{T}},
                                              Y::AbstractVecOrMat{Complex{T}})
    ntrials, nchX = size(X)
    nchY = size(Y, 2)
    Xsurrogate, Ysurrogate, outtmp, worktrue, worksurrogate = work
    trueval = out.trueval
    surrogates = out.surrogates
    stat = t.transform

    (size(Xsurrogate, 1) == ntrials - 1 && size(Xsurrogate, 2) == nchX &&
     size(Ysurrogate, 1) == ntrials - 1 && size(Ysurrogate, 2) == nchY) || error("invalid work object")
    ntrials > 0 || error("X is empty")
    chkinput(trueval, X, Y)

    fill!(surrogates, NaN)

    # True value
    computestat!(stat, trueval, worktrue, X, Y)

    # First jackknife
    copyrows!(Xsurrogate, 1, X, 2:size(X, 1))
    copyrows!(Ysurrogate, 1, Y, 2:size(Y, 1))
    computestat!(stat, outtmp, worksurrogate, Xsurrogate, Ysurrogate)
    surrogates[1, :, :] = outtmp

    # Subsequent jackknifes
    for i = 2:ntrials
        copyrows!(Xsurrogate, i-1, X, i-1)
        copyrows!(Ysurrogate, i-1, Y, i-1)
        computestat!(stat, outtmp, worksurrogate, Xsurrogate, Ysurrogate)
        surrogates[i, :, :] = outtmp
    end

    out
end

# Estimate of variance from jackknife surrogates
function jackknife_var!{T}(out::Matrix{T}, jn::JackknifeOutput)
    surrogates = jn.surrogates
    n = size(surrogates, 1)

    for k = 1:size(surrogates, 3), j = 1:size(surrogates, 2)
        # Compute mean
        m = zero(eltype(surrogates))
        @simd for i = 1:n
            @inbounds m += surrogates[i, j, k]
        end
        m /= n

        # Compute variance
        v = zero(eltype(out))
        @simd for i = 1:n
            @inbounds v += abs2(surrogates[i, j, k] - m)
        end

        @inbounds out[j, k] = v*(n-1)/n
    end
    out
end
jackknife_var(jn::JackknifeOutput) = jackknife_var!(similar(jn.trueval), jn)

# Estimate of (first-order) bias from jackknife surrogates
function jackknife_bias!(out::Matrix, jn::JackknifeOutput)
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
jackknife_bias(jn::JackknifeOutput) = jackknife_bias!(similar(jn.trueval), jn)

# Jackknifing of n-d arrays
function computestat!{S,T<:Complex}(t::Jackknife, out::JackknifeOutput, work::S, X::AbstractArray{T})
    trueval = out.trueval
    surrogates = out.surrogates
    !isempty(X) || error(ArgumentError("X is empty"))
    for i = 1:Base.trailingsize(X, 3)
        computestat!(t, JackknifeOutput(sub(trueval, :, :, i), sub(surrogates, :, :, :, i)),
                     work, sub(X, :, :, i))
    end
    out
end

#
# Inefficient generic bootstrap computation
#

immutable Bootstrap{R<:Statistic} <: Statistic
    transform::R
    weights::Matrix{Int32}    # Sample weights for each bootstrap. nbootstraps x ntrials
                              # This is transposed for efficiency when weighting output

    function Bootstrap(transform, weights)
        any(sum(weights, 2) .!= size(weights, 2)) && throw(ArgumentError("weights must sum to ntrials"))
        new(transform, weights)
    end
end
Bootstrap(transform::Statistic, weights) = Bootstrap{typeof(transform)}(transform, weights)

for n = 2:6
    @eval begin
        allocoutput{T<:Complex}(t::Bootstrap, X::AbstractArray{T,$n}, Y::AbstractArray{T,$n}=X) =
            zeros(eltype(t.transform, X, Y), size(t.weights, 1), size(X, 2), size(Y, 2), $([:(size(X, $i)) for i = 3:n]...))
    end
end

#genweights(ntrials::Int, nbootstraps::Int) = rand(Multinomial(ntrials, ntrials), nbootstraps)'
function genweights(ntrials::Int, nbootstraps::Int)
    weights = zeros(Int32, ntrials, nbootstraps)
    rnd = zeros(Int32, nbootstraps)
    for iboot = 1:nbootstraps
        @compat rand!(rnd, Int32(1):Int32(ntrials))
        for itrial = 1:ntrials
            @inbounds weights[rnd[itrial], iboot] += @compat Int32(1)
        end
    end
    weights'
end
Bootstrap(transform::Statistic, ntrials::Int, nbootstraps::Int) =
    Bootstrap(transform, genweights(ntrials, nbootstraps))

function copybootstraps!(out, X, weights, ibootstrap)
    @inbounds for ichannel = 1:nchannels(out)
        n = 1
        for itrial = 1:size(weights, 1)
            fin = n + weights[itrial, ibootstrap]
            v = X[itrial, ichannel]
            while n < fin
                out[n, ichannel] = v
                n += 1
            end
        end
    end
    out
end

allocwork{T<:Real}(t::Bootstrap, X::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), t.weights', allocoutput(t.transform, X), allocwork(t.transform, X))
function computestat!{R<:Statistic,T<:Real,V}(t::Bootstrap{R}, out::AbstractArray{V,3},
                                              work::(Matrix{Complex{T}}, Matrix{Int32}, Matrix{V}, Any),
                                              X::AbstractVecOrMat{Complex{T}})
    stat = t.transform
    ntrials, nch = size(X)
    Xbootstrap, weights, bsoutput, bswork = work

    (size(X, 1) == size(Xbootstrap, 1) == size(weights, 1) &&
     size(X, 2) == size(Xbootstrap, 2)) || throw(ArgumentError("invalid work object"))
    chkinput(bsoutput, X)

    for ibootstrap = 1:size(weights, 2)
        copybootstraps!(Xbootstrap, X, weights, ibootstrap)
        computestat!(stat, bsoutput, bswork, Xbootstrap)
        out[ibootstrap, :, :] = bsoutput
    end

    out
end

allocwork{T<:Real}(t::Bootstrap, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Complex{T}, size(X, 1), size(X, 2)), Array(Complex{T}, size(Y, 1), size(Y, 2)),
     t.weights', allocoutput(t.transform, X, Y), allocwork(t.transform, X, Y))
function computestat!{R<:Statistic,T<:Real,V}(t::Jackknife{R}, out::AbstractArray{V,3},
                                              work::(Matrix{Complex{T}}, Matrix{Complex{T}}, Matrix{Int32}, Matrix{V}, Any),
                                              X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    ntrials, nchX = size(X)
    nchY = size(Y, 2)
    Xbootstrap, Ybootstrap, weights, bsoutput, bswork = work

    (size(X, 1) == size(Xbootstrap, 1) == size(weights, 1) == size(out, 1) &&
     size(Y, 1) == size(Ybootstrap, 1) == size(weights, 1) == size(out, 1) &&
     size(X, 2) == size(Xbootstrap, 2) &&
     size(Y, 2) == size(Ybootstrap, 2)) || throw(ArgumentError("invalid work object"))
    chkinput(bsoutput, X, Y)

    for ibootstrap = 1:size(weights, 2)
        copybootstraps!(Xbootstrap, X, weights, ibootstrap)
        copybootstraps!(Ybootstrap, Y, weights, ibootstrap)
        computestat!(stat, bsoutput, bswork, Xbootstrap, Ybootstrap)
        out[ibootstrap, :, :] = bsoutput
    end

    out
end


#
# Transforms
#

include("transforms/spectrum.jl")
include("transforms/coherence.jl")
include("transforms/plv.jl")
include("transforms/pli.jl")
include("transforms/jammalamadaka.jl")
include("transforms/juppmardia.jl")
include("transforms/modulationindex.jl")

export computestat, computestat!, PowerSpectrum, CrossSpectrum, Coherency,
       Coherence, MeanPhaseDiff, PLV, PPC, PLI, PLI2Unbiased, WPLI,
       WPLI2Debiased, JammalamadakaR, JuppMardiaR,
       HurtadoModulationIndex, Jackknife, jackknife_bias, jackknife_var

#
# Utilities
#

# Upper triangular symmetric/hermitian multiply, ignoring lower triangular part
A_mul_Ac!(out::AbstractMatrix, X::AbstractMatrix) = A_mul_Bc!(out, A, A)
A_mul_Ac!{T}(out::StridedMatrix{T}, X::StridedMatrix{T}) =
    BLAS.herk!('U', 'N', one(T), X, zero(T), out)
A_mul_At!(out::AbstractMatrix, X::AbstractMatrix) = A_mul_Bt!(out, A, A)
A_mul_At!{T}(out::StridedMatrix{T}, X::StridedMatrix{T}) =
    BLAS.syrk!('U', 'N', one(T), X, zero(T), out)

# Scale and conjugate
function conjscale!(X::AbstractArray, s::Number)
    for i = 1:length(X)
        @inbounds X[i] = conj(X[i])*s
    end
    X
end
function conjscale!(out::AbstractArray, X::AbstractArray, s::Number)
    for i = 1:length(X)
        @inbounds out[i] = conj(X[i])*s
    end
    X
end

# Normalize to unit circle
function unitnormalize!{T<:Complex,N}(out::AbstractArray{T,N}, X::AbstractArray{T,N})
    length(out) == length(X) || error(DimensionMismatch("work"))
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
        error(DimensionMismatch("out"))

    for i = 1:size(XXc, 2)
        @inbounds out[i, i] = 1/sqrt(real(XXc[i, i]))
    end
    for j = 1:size(XXc, 2)
        x = real(out[j, j])
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
                           X::AbstractMatrix, Y::AbstractMatrix,
                           Xtmp::AbstractVector{T}, Ytmp::AbstractVector{T},
                           XYc::Union(AbstractMatrix{Complex{T}}, AbstractMatrix{T})=out,
                           f=Base.IdFun())
    (size(XYc, 1) == length(Xtmp) && size(XYc, 2) == length(Ytmp)) ||
        error(DimensionMismatch("work"))
    (size(out, 1) == size(XYc, 1) && size(out, 2) == size(XYc, 2)) ||
        error(DimensionMismatch("out"))

    sumabs2!(Xtmp, X)
    sumabs2!(Ytmp, Y)
    for i = 1:length(Xtmp)
        Xtmp[i] = 1/sqrt(Xtmp[i])
    end
    for i = 1:length(Xtmp)
        Ytmp[i] = 1/sqrt(Ytmp[i])
    end

    for j = 1:size(XYc, 2), i = 1:size(XYc, 1)
        @inbounds out[i, j] = f(XYc[i, j])*(Xtmp[i]*Ytmp[j])
    end
    out
end

# Fastest possible unsafe submatrix
immutable UnsafeSubMatrix{T<:Number} <: DenseMatrix{T}
    ptr::Ptr{T}
    nrow::Int
    ncol::Int
end
Base.getindex(X::UnsafeSubMatrix, i::Int) = unsafe_load(X.ptr, i)
Base.getindex(X::UnsafeSubMatrix, i::Int, j::Int) = unsafe_load(X.ptr, x.nrow*(j-1)+i)
Base.setindex!{T}(X::UnsafeSubMatrix{T}, x, i::Int) = unsafe_store!(X.ptr, convert(T, x), i)
Base.setindex!{T}(X::UnsafeSubMatrix{T}, x, i::Int, j::Int) = unsafe_store!(X.ptr, convert(T, x), x.nrow*(j-1)+i)
Base.pointer(X::UnsafeSubMatrix) = X.ptr
Base.convert{T}(::Type{Ptr{T}}, X::UnsafeSubMatrix) = X.ptr
Base.size(X::UnsafeSubMatrix) = X.nrow, X.ncol

immutable ConjFun <: Base.Func{1} end
call(::ConjFun, x) = conj(x)

#
# Types and basics
#

abstract Statistic

abstract PairwiseStatistic <: Statistic
abstract RealPairwiseStatistic <: PairwiseStatistic
abstract ComplexPairwiseStatistic <: PairwiseStatistic
Base.eltype{T<:Real}(::RealPairwiseStatistic, X::AbstractArray{Complex{T}}) = T
Base.eltype{T<:Real}(::ComplexPairwiseStatistic, X::AbstractArray{Complex{T}}) = Complex{T}
Base.eltype{T<:Real,S<:Real}(::RealPairwiseStatistic, X::AbstractArray{Complex{T}}, Y::AbstractArray{Complex{S}}) =
    promote_type(T, S)
Base.eltype{T<:Real,S<:Real}(::ComplexPairwiseStatistic, X::AbstractArray{Complex{T}}, Y::AbstractArray{Complex{S}}) =
    Complex{promote_type(T, S)}
allocoutput{T<:Real}(t::PairwiseStatistic, X::AbstractMatrix{Complex{T}}) =
    zeros(eltype(t, X), size(X, 1), size(X, 1))
allocoutput{T<:Real}(t::PairwiseStatistic, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    zeros(eltype(t, X, Y), size(X, 1), size(Y, 1))

# X is channels x trials
# output is channels x channels (upper triangular)
computestat(t::Statistic, X::AbstractArray) =
    computestat!(t, allocoutput(t, X), allocwork(t, X), X)
computestat(t::Statistic, X::AbstractArray, Y::AbstractArray) =
    computestat!(t, allocoutput(t, X, Y), allocwork(t, X, Y), X, Y)

# X is channels x trials x ...
# output is channels x channels x ...
allocwork(t, X::AbstractArray) = allocwork(t, UnsafeSubMatrix(pointer(X), size(X, 1), size(X, 1)))
allocoutput{T<:Complex}(t, X::AbstractArray{T,3}) =
    zeros(eltype(t, X), size(X, 1), size(X, 1), size(X, 3))
allocoutput{T<:Complex}(t, X::AbstractArray{T,4}) =
    zeros(eltype(t, X), size(X, 1), size(X, 1), size(X, 3), size(X, 4))
allocoutput{T<:Complex}(t, X::AbstractArray{T,5}) =
    zeros(eltype(t, X), size(X, 1), size(X, 1), size(X, 3), size(X, 4), size(X, 5))
allocoutput{T<:Complex}(t, X::AbstractArray{T,6}) =
    zeros(eltype(t, X), size(X, 1), size(X, 1), size(X, 3), size(X, 4), size(X, 5), size(X, 6))
function computestat!{S,T<:Complex}(t::Statistic, out::AbstractArray, work::S, X::AbstractArray{T})
    !isempty(X) || error(ArgumentError("X is empty"))
    lead = size(X, 1)*size(X, 2)
    leadout = size(X, 1)*size(X, 1)
    trail = Base.trailingsize(X, 3)
    Base.trailingsize(out, 3) == trail || error(DimensionMismatch("out"))

    sz = (size(X, 1), size(X, 2))
    szout = (size(X, 1), size(X, 1))
    for i = 1:trail
        computestat!(t, UnsafeSubMatrix(pointer(out, leadout*(i-1)+1), szout...),
                     work, UnsafeSubMatrix(pointer(X, lead*(i-1)+1), sz...))
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
# Generic jackknife surrogate computation
#

immutable Jackknife{R<:Statistic} <: Statistic
    transform::R
end
immutable JackknifeOutput{T<:Array,S<:Array}
    trueval::T
    surrogates::S
end
function allocwork{T<:Real}(t::Jackknife, X::AbstractMatrix{Complex{T}})
    Xsurrogate = Array(Complex{T}, size(X, 1), size(X, 2)-1)
    (Xsurrogate, allocwork(t.transform, X), allocwork(t.transform, Xsurrogate))
end
function computestat!{R<:Statistic,T<:Real,V}(t::Jackknife{R}, out::JackknifeOutput,
                                              work::(Matrix{Complex{T}}, V, V),
                                              X::AbstractMatrix{Complex{T}})
    stat = t.transform
    nch, ntrials = size(X)
    Xsurrogate, worktrue, worksurrogate = work
    trueval = out.trueval
    surrogates = out.surrogates
    fill!(surrogates, NaN)

    # True value
    computestat!(stat, trueval, worktrue, X)

    # First jackknife
    copy!(Xsurrogate, 1, X, nch+1, (ntrials-1)*nch)
    surrogateout = UnsafeSubMatrix(pointer(surrogates), nch, nch)
    computestat!(stat, surrogateout, worksurrogate, Xsurrogate)

    # Subsequent jackknifes
    for i = 2:ntrials
        copy!(Xsurrogate, (i-2)*nch+1, X, (i-2)*nch+1, nch)
        surrogateout = UnsafeSubMatrix(pointer(surrogates, (i-1)*nch*nch+1), nch, nch)
        computestat!(stat, surrogateout, worksurrogate, Xsurrogate)
    end

    out
end

# Estimate of variance from jackknife surrogates
function jackknife_var!{T}(out::Matrix{T}, work::Matrix{T}, jn::JackknifeOutput)
    surrogates = jn.surrogates
    n = size(surrogates, 3)
    mean!(work, surrogates)
    scale!(Base.varm(surrogates, work, 3; corrected=false), n-1)
end
jackknife_var(jn::JackknifeOutput) = jackknife_var!(similar(jn.trueval), similar(jn.trueval), jn)

# Estimate of (first-order) bias from jackknife surrogates
function jackknife_bias!(out::Matrix, jn::JackknifeOutput)
    trueval = jn.trueval
    surrogates = jn.surrogates
    n = size(surrogates, 3)

    sum!(out, surrogates)
    for i = 1:length(out)
        out[i] = (n - 1)*(out[i]/n - trueval[i])
    end
    out
end
jackknife_bias(jn::JackknifeOutput) = jackknife_bias!(similar(jn.trueval), jn)

# Jackknifing of n-d arrays
allocoutput{T<:Real}(t::Jackknife, X::AbstractMatrix{Complex{T}}) =
    JackknifeOutput(zeros(eltype(t.transform, X), size(X, 1), size(X, 1)),
                    zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 2)))
allocoutput{T<:Real}(t::Jackknife, X::AbstractArray{Complex{T},3}) =
    JackknifeOutput(zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 3)),
                    zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 2), size(X, 3)))
allocoutput{T<:Real}(t::Jackknife, X::AbstractArray{Complex{T},4}) =
    JackknifeOutput(zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 3), size(X, 4)),
                    zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 2), size(X, 3), size(X, 4)))
allocoutput{T<:Real}(t::Jackknife, X::AbstractArray{Complex{T},5}) =
    JackknifeOutput(zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 3), size(X, 4), size(X, 5)),
                    zeros(eltype(t.transform, X), size(X, 1), size(X, 1), size(X, 2), size(X, 3), size(X, 4), size(X, 5)))
function computestat!{S,T<:Complex}(t::Statistic, out::JackknifeOutput, work::S, X::AbstractArray{T})
    trueval = out.trueval
    surrogates = out.surrogates
    !isempty(X) || error(ArgumentError("X is empty"))
    lead = size(X, 1)*size(X, 2)
    leadtrue = size(X, 1)*size(X, 1)
    leadsurrogates = leadtrue*size(X, 2)
    trail = Base.trailingsize(X, 3)

    sz = (size(X, 1), size(X, 2))
    sztrueval = (size(X, 1), size(X, 1))
    szsurrogates = (size(X, 1), size(X, 1), size(X, 2))
    for i = 1:trail
        computestat!(t, JackknifeOutput(pointer_to_array(pointer(trueval, leadtrue*(i-1)+1), sztrueval),
                                        pointer_to_array(pointer(surrogates, leadsurrogates*(i-1)+1), szsurrogates)),
                     work, pointer_to_array(pointer(X, lead*(i-1)+1), sz))
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

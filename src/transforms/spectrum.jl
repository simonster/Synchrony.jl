#
# Power spectrum
#

immutable PowerSpectrum <: Statistic; end

Base.eltype{T<:Real}(::PowerSpectrum, X::AbstractMatrix{Complex{T}}) = T
allocwork{T<:Complex}(::PowerSpectrum, X::AbstractMatrix{Complex{T}}) = nothing
allocoutput{T<:Real}(::PowerSpectrum, X::AbstractMatrix{Complex{T}}) =
    Array(T, nchannels(X), 1)

# Single input matrix
computestat!{T<:Real}(::PowerSpectrum, out::AbstractMatrix{T}, ::Nothing,
                      X::AbstractMatrix{Complex{T}}) =
    scale!(sumabs2!(out, X), 1/size(X, 2))

#
# Cross spectrum
#

immutable CrossSpectrum <: ComplexPairwiseStatistic; end

# Single input matrix
allocwork{T<:Complex}(::CrossSpectrum, X::AbstractMatrix{T}) = nothing
computestat!{T<:Complex}(::CrossSpectrum, out::AbstractMatrix{T}, ::Nothing,
                         X::AbstractMatrix{T}) =
    conjscale!(A_mul_Ac!(out, X), 1/size(X, 2))

# Two input matrices
allocwork{T<:Complex}(::CrossSpectrum, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) = nothing
computestat!{T<:Complex}(::CrossSpectrum, out::AbstractMatrix{T}, ::Nothing,
                         X::AbstractMatrix{T}, Y::AbstractMatrix{T}) =
    conjscale!(A_mul_Bc!(out, X, Y), 1/size(X, 2))

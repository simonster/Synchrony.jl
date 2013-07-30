module Multitaper
using NumericExtensions

export dpss, psd, xspec, coherence, mtrfft

# Compute discrete prolate spheroid sequences (Slepian tapers)
function dpss(n::Int, nw::Real, ntapers::Int=iceil(2*nw)-1)
    # Construct symmetric tridiagonal matrix
    # See Gruenbacher, D. M., & Hummels, D. R. (1994). A simple algorithm
    # for generating discrete prolate spheroidal sequences. IEEE
    # Transactions on Signal Processing, 42(11), 3276-3278.
    i1 = 0:(n-1)
    i2 = 1:(n-1)
    mat = SymTridiagonal(cos(2pi*nw/n)*((n - 1)/2 - i1).^2, 0.5.*(i2*n - i2.^2))

    # Get tapers
    ev = eigvals(mat, n-ntapers+1, n)
    v = fliplr(eigvecs(mat, ev)[1])

    # Polarity convention is that first lobe of DPSS is positive
    sgn = ones(size(v, 2))
    sgn[2:2:end] = sign(v[2, 2:2:end])
    scale!(v, sgn)
end

#
# Basic functionality, for single series and pairs
#
# Estimate power spectral density
function psd{T<:Real,N}(A::AbstractArray{T,N}; tapers::Matrix=dpss(size(A, 1), 4),
                        pad::Union(Bool, Int)=true, fs::Real=1.0)
    n = size(A, 1)
    nfft = getpadding(size(A, 1), pad)
    tmp = zeros(complextype(eltype(A)), nfft, prod(size(A)[2:end])::Int)
    out = zeros(eltype(A), div(nfft, 2)+1, size(A)[2:end]...)

    nout = size(out, 1)
    zerorg = n+1:nfft
    ntapers = size(tapers, 2)
    perform_fft! = plan_fft!(tmp, 1)

    for i = 1:ntapers
        for j = 1:div(length(A), n)
            Aoff = (j-1)*n
            for k = 1:n
                tmp[k, j] = A[Aoff+k].*tapers[k, i]
            end
            tmp[zerorg, j] = zero(eltype(tmp))
        end
        
        perform_fft!(tmp)

        for j = 1:div(length(out), nout)
            outoff = (j-1)*nout
            for k = 1:nout
                out[outoff+k] += abs2(tmp[k, j])
            end
        end
    end
    scalespectrum!(out, nfft, fs*ntapers)
end

# Estimate cross-spectrum
function xspec{T<:Real,N}(A::AbstractArray{T,N}, B::AbstractArray{T,N},
                          accumulator::BinaryFunctor=Dot();
                          tapers::Matrix=dpss(size(A, 1), 4),
                          pad::Union(Bool, Int)=true, fs::Real=1.0)
    if size(A) != size(B)
        throw(Base.DimensionMismatch("A and B must be the same size"))
    end
    n = size(A, 1)
    nfft = getpadding(size(A, 1), pad)
    ffttype = complextype(eltype(A))
    tmp = zeros(ffttype, nfft, prod(size(A)[2:end])::Int, 2)
    sXY = zeros(result_type(accumulator, ffttype, ffttype), div(nfft, 2)+1, size(A)[2:end]...)
    sXX = zeros(eltype(A), div(nfft, 2)+1, size(A)[2:end]...)
    sYY = zeros(eltype(A), div(nfft, 2)+1, size(A)[2:end]...)

    n = size(A, 1)
    nfft = size(tmp, 1)
    nout = size(sXY, 1)
    zerorg = n+1:nfft
    ntapers = size(tapers, 2)
    perform_fft! = plan_fft!(tmp, 1)

    for i = 1:ntapers
        for j = 1:div(length(A), n)
            Aoff = (j-1)*n
            for k = 1:n
                tmp[k, j, 1] = A[Aoff+k].*tapers[k, i]
                tmp[k, j, 2] = B[Aoff+k].*tapers[k, i]
            end
            tmp[zerorg, j, :] = zero(eltype(tmp))
        end
        
        perform_fft!(tmp)
        scalespectrum!(tmp, nfft, fs*ntapers, true)

        for j = 1:div(length(sXY), nout)
            outoff = (j-1)*nout
            for k = 1:nout
                outind = outoff+k
                sXY[outind] += evaluate(accumulator, tmp[k, j, 1], tmp[k, j, 2])
                sXX[outind] += abs2(tmp[k, j, 1])
                sYY[outind] += abs2(tmp[k, j, 2])
            end
        end
    end
    (sXY, sXX, sYY)
end

#
# Functionality for multiple channels
#
# Get all pairs of channels
function allpairs(n)
    combs = Array((Int, Int), binomial(n, 2))
    k = 0
    for i = 1:n-1, j = i+1:n
        combs[k += 1] = (i, j)
    end
    combs
end

# Perform tapered FFT along first dimension
function mtrfft{T<:Real,N}(A::AbstractArray{T,N}; tapers::Matrix=dpss(size(A, 1), 4),
                           pad::Union(Bool, Int)=true, fs::Real=1.0)
    nfft = getpadding(size(A, 1), pad)
    n = size(A, 1)
    sz = size(A)

    X = Array(eltype(A), nfft, sz[2:end]..., size(tapers, 2))::Array{T,N+1}
    Xstride = stride(X, N+1)
    for i = 1:size(tapers, 2), j = 1:div(length(A), n)
        Aoff = (j-1)*n
        Xoff = (j-1)*nfft+Xstride*(i-1)
        for k = 1:n
            X[Xoff+k] = A[Aoff+k].*tapers[k, i]
        end
        X[Xoff+(n+1:nfft)] = zero(eltype(X))
    end
    
    scalespectrum!(rfft(X, 1), nfft, fs*size(tapers, 2), true)
end

# Estimate power spectra
# A is output of mtrfft (time x trials x channels x tapers)
psd{T<:Real}(X::AbstractArray{Complex{T}, 4}) = squeeze(sqsum(X, 4), 4)::Array{T, 3}

# Estimate cross-spectrum
# A is output of mtrfft (time x trials x channels x tapers)
function xspec{T<:Complex}(X::AbstractArray{T, 4}, pairs::AbstractVector{(Int, Int)},
                           accumulator::BinaryFunctor=Dot(); trialavg::Bool=false)
    xs = zeros(result_type(accumulator, eltype(X), eltype(X)), size(X, 1), trialavg ? 1 : size(X, 2), length(pairs))
    if trialavg
        for l = 1:size(X, 4), k = 1:length(pairs)
            ch1, ch2 = pairs[k]
            for j = 1:size(X, 2), i = 1:size(X, 1)
                xs[i, 1, k] += evaluate(accumulator, X[i, j, ch1, l], X[i, j, ch2, l])
            end
        end
    else
        for l = 1:size(X, 4), k = 1:length(pairs)
            ch1, ch2 = pairs[k]
            for j = 1:size(X, 2), i = 1:size(X, 1)
                xs[i, j, k] += evaluate(accumulator, X[i, j, ch1, l], X[i, j, ch2, l])
            end
        end
    end
    xs
end

#
# Measures of synchrony
#

macro synchronyop(fn, accumulator, code)
    # Only coherence uses the PSD
    computepsd = fn == :coherence
    esc(quote
        function $fn{T<:Complex}(A::AbstractArray{T, 4}, pairs::AbstractVector{(Int, Int)};
                                 trialavg::Bool=false)
            n = trialavg ? size(A, 2)*size(A, 4) : size(A, 4)
            xs = xspec(A, pairs, $(accumulator)(); trialavg=trialavg)
            $(if computepsd
                :(s = psd(A))
            end)
            for k = 1:length(pairs), j = 1:size(xs, 2), i = 1:size(xs, 1)
                ch1, ch2 = pairs[k]
                x = xs[i, j, k]
                $(if computepsd
                    :((sXX, sYY) = (s[i, j, ch1], s[i, j, ch2]))
                end)
                xs[i, j, k] = $code
            end
            xs
        end

        function $fn{T<:Real,N}(A::AbstractArray{T,N}, B::AbstractArray{T,N};
                                tapers::Matrix=dpss(size(A, 1), 4),
                                pad::Union(Bool, Int)=true, fs::Real=1.0)
            n = size(tapers, 2)
            sXYa, sXXa, sYYa = xspec(A, B, $(accumulator)(); tapers=tapers, pad=pad, fs=fs)
            for i = 1:length(sXYa)
                x = sXYa[i]
                $(if computepsd
                    :((sXX, sYY) = (sXXa[i], sYYa[i]))
                end)
                sXYa[i] = $code
            end
            sXYa
        end
    end)
end

abstract ConnectivityAccumulator <: BinaryFunctor
result_type(::ConnectivityAccumulator, x, y) = promote_type(x, y)

# Connectivity indices are defined in terms of two map functions. The first is a
# NumericFunctor of type ConnectivityAccumulator. The second is a code fragment in
# a macro.

# Coherence
type Dot <: ConnectivityAccumulator; end
evaluate(::Dot, x, y) = dot(x, y)
@synchronyop coherence Dot begin
    x/sqrt(sXX*sYY)
end

# Phase locking value (imaginary) and pairwise phase consistence (unbiased PLV)
type NormalizedDot <: BinaryFunctor; end
function evaluate(::NormalizedDot, x, y)
    z = dot(x, y)
    z/abs(z)
end
@synchronyop plv NormalizedDot begin
    x/n
end
@synchronyop ppc NormalizedDot begin
    (dot(x, x) - n)/(n * (n - 1))
end

# PLI (signed) and unbiased PLI^2
type ImDotSign <: BinaryFunctor; end
function evaluate(::ImDotSign, x, y)
    z = imag(dot(x, y))
    z > 0 ? 1 : z < 0 ? -1 : 0
end
result_type(::ImDotSign, x, y) = Int
@synchronyop pli ImDotSign begin
    x/n
end
@synchronyop pli2_unbiased ImDotSign begin
    (n * abs2(x/n) - 1)/(n - 1)
end

# Weighted PLI (signed)
# This is a hack. We need to accumulate two reals, so we put them in a complex number
type WPLIAccumulator <: BinaryFunctor; end
function evaluate(::WPLIAccumulator, x, y)
    z = imag(dot(x, y))
    complex(z, abs(z))
end
@synchronyop wpli WPLIAccumulator begin
    real(x)/imag(x)
end

#
# Helper functions
#
getpadding(n::Int, padparam::Bool) = padparam ? nextpow2(n) : n
getpadding(n::Int, padparam::Int) = padparam

# Scale real part of spectrum to satisfy Parseval's theorem
# If sq is true, scale by sqrt of values (for FFTs)
function scalespectrum!(spectrum::AbstractArray, n::Int, divisor::Real, sq::Bool=false)
    d = 2/divisor
    if sq; d = sqrt(d); end
    scale!(spectrum, d)

    s = size(spectrum, 1)
    for i = 1:div(length(spectrum), s)
        off = (i-1)*s+1
        spectrum[off] /= sq ? sqrt(2) : 2
        if iseven(n)
            spectrum[off+div(n, 2)] /= sq ? sqrt(2) : 2
        end
    end
    spectrum
end

# Get the equivalent complex type for a given type
complextype{T<:Complex}(::Type{T}) = T
complextype{T}(::Type{T}) = Complex{T}
end
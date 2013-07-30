module Multitaper
using NumericExtensions

export PowerSpectrum, CrossSpectrum, Coherence, dpss, multitaper, psd, xspec, coherence, allpairs, frequencies

#
# Statistics computed on transformed data
#
abstract TransformStatistic{T<:Real}

# Convenience functions for pairwise statistics
abstract PairwiseTransformStatistic{T<:Real} <: TransformStatistic{T}
macro pairwisestat(name, xtype)
    esc(quote
        type $name{T<:Real} <: PairwiseTransformStatistic{T}
            pairs::Vector{(Int, Int)}
            x::Array{$xtype, 2}
            $name() = new()
            $name(pairs::Vector{(Int, Int)}) = new(pairs)
        end
        $name() = $name{Float64}()
    end)
end
function init{T}(s::PairwiseTransformStatistic{T}, nout, nchannels, nsamples)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.x = zeros(eltype(fieldtype(s, :x)), nout, length(s.pairs))
end
macro accumulatebypair(t, j, i, x, y, code)
    quote
        function $(esc(:accumulate))(s::$t, fftout)
            pairs = s.pairs
            for $i = 1:length(pairs)
                ch1, ch2 = pairs[$i]
                for $j = 1:size(fftout, 1)
                    $x = fftout[$j, ch1]
                    $y = fftout[$j, ch2]
                    $code
                end
            end
        end
    end
end

# Power Spectrum
type PowerSpectrum{T<:Real} <: TransformStatistic{T}
    x::Array{T, 2}
    PowerSpectrum() = new()
end
PowerSpectrum() = PowerSpectrum{Float64}()
function init{T}(s::PowerSpectrum{T}, nout, nchannels, nsamples)
    s.x = zeros(T, nout, nchannels)
end
function accumulate(s::PowerSpectrum, fftout)
    A = s.x
    for i = 1:length(fftout)
        A[i] += abs2(fftout[i])
    end
end
finish(s::PowerSpectrum, nsamples) = scale!(s.x, 1/nsamples)

# Cross Spectrum
@pairwisestat CrossSpectrum Complex{T}
@accumulatebypair CrossSpectrum j i x y begin
    s.x[j, i] += conj(x)*y
end
finish(s::CrossSpectrum, nsamples) = scale!(s.x, 1/nsamples)

# Coherence
type Coherence{T<:Real} <: PairwiseTransformStatistic{T}
    pairs::Vector{(Int, Int)}
    psd::PowerSpectrum{T}
    xspec::CrossSpectrum{T}
    Coherence() = new()
    Coherence(pairs::Vector{(Int, Int)}) = new(pairs)
end
Coherence() = Coherence{Float64}()
function init{T}(s::Coherence{T}, nout, nchannels, nsamples)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.psd = PowerSpectrum{T}()
    s.xspec = CrossSpectrum{T}(s.pairs)
    init(s.psd, nout, nchannels, nsamples)
    init(s.xspec, nout, nchannels, nsamples)
end
function accumulate(s::Coherence, fftout)
    accumulate(s.psd, fftout)
    accumulate(s.xspec, fftout)
end
function finish(s::Coherence, nsamples)
    psd = finish(s.psd, nsamples)
    xspec = finish(s.xspec, nsamples)
    pairs = s.pairs
    for i = 1:length(pairs)
        ch1, ch2 = pairs[i]
        for j = 1:size(xspec, 1)
            xspec[j, i] = xspec[j, i]/sqrt(psd[j, ch1]*psd[j, ch2])
        end
    end
    xspec
end

#
# Core functionality
#
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

# Compute frequencies based on length of input
frequencies(nfft::Int, fs::Real=1.0) = (0:div(nfft, 2))*(fs/nfft)
function frequencies(nfft::Int, fs::Real, fmin::Real, fmax::Real=Inf)
    allfreq = frequencies(nfft, fs)
    allfreq[searchsortedfirst(allfreq, fmin):(fmax == Inf ? length(allfreq) : searchsortedlast(allfreq, fmax))]
end
frequencies{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                     fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf) =
    frequencies(nextpow2(size(A, 1)), fs, fmin, fmax)

# Perform tapered FFT along first dimension for each channel along second dimension,
# accumulating along third dimension
function multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                             stats::(TransformStatistic, TransformStatistic...);
                             tapers::Matrix=dpss(size(A, 1), 4), nfft::Int=nextpow2(size(A, 1)),
                             fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf)
    allfreq = frequencies(nfft, fs)
    frange = searchsortedfirst(allfreq, fmin):(fmax == Inf ? length(allfreq) : searchsortedlast(allfreq, fmax))

    n = size(A, 1)
    nout = nfft >> 1 + 1
    zerorg = n+1:nfft
    ntapers = size(tapers, 2)
    nchannels = size(A, 2)
    nsamples = size(A, 3)*ntapers
    multiplier = sqrt(2/fs)

    for stat in stats
        init(stat, length(frange), nchannels, nsamples)
    end

    fftin = Array(T, nfft, size(A, 2))
    fftout = Array(Complex{T}, nout, size(A, 2))
    #fftview = sub(fftout, frange, 1:size(A,2))
    fftview = zeros(Complex{T}, length(frange), size(A, 2))

    p = FFTW.Plan(fftin, fftout, 1, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)
    scalelast = iseven(nfft) && frange[end] == length(allfreq)

    for i = 1:ntapers, j = 1:size(A, 3)
        for k = 1:nchannels
            for l = 1:n
                fftin[l, k] = A[l, k, j].*tapers[l, i]
            end
            for l = n+1:nfft
                fftin[l, k] = zero(T)
            end
        end

        # Exectute rfft with output preallocated
        FFTW.execute(T, p.plan)

        # Scale output to conform to Parseval's theorem
        for k = 1:nchannels
            for l = 1:length(frange)
                fftview[l, k] = fftout[frange[l], k]*multiplier
            end
            if frange[1] == 1
                fftview[1, k] /= sqrt(2)
            end
            if scalelast
                fftview[end, k] /= sqrt(2)
            end
        end

        # Accumulate
        for stat in stats
            accumulate(stat, fftview)
        end
    end

    [finish(stat, nsamples) for stat in stats]
end

# Calling with types instead of instances
multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                    stat::(DataType, DataType...); tapers::Matrix=dpss(size(A, 1), 4),
                    nfft::Int=nextpow2(size(A, 1)), fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf) =
    multitaper(A, tuple([x() for x in stat]...); tapers=tapers, nfft=nfft, fs=fs, fmin=fmin, fmax=fmax)

# Calling with a single type or instance
multitaper{T<:Real,S<:TransformStatistic}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                                          stat::Union(S, Type{S}); tapers::Matrix=dpss(size(A, 1), 4),
                                          nfft::Int=nextpow2(size(A, 1)), fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf) =
    multitaper(A, (stat,); tapers=tapers, nfft=nfft, fs=fs, fmin=fmin, fmax=fmax)[1]

#
# Convenience functions
#
# Estimate power spectral density
psd{T<:Real}(A::AbstractArray{T}; kw...) =
    multitaper(A, (PowerSpectrum{T}(),); kw...)[1]
xspec{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (CrossSpectrum{T}(),); kw...)[1]
coherence{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (Coherence{T}(),); kw...)[1]

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
import Base: getindex, size, ndims, convert
export MorletWavelet, MorseWavelet, wavebases, fstd, tstd, ContinuousWaveletTransform, cwt

#
# Mother wavelets, which are convolved with the signal in frequency space
# 
abstract MotherWavelet{T<:Real}

# The analytic Morlet wavelet. See:
# Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet
# analysis. Bulletin of the American Meteorological Society, 79(1),
# 61–78.
#
# freq is the frequency in Hz
# k0 is the wave number
immutable MorletWavelet{T} <: MotherWavelet{T}
    freq::Vector{T}
    k0::T
    fourierfactor::T
end
MorletWavelet{T<:Real}(freq::Vector{T}, k0::Real=5.0) =
    MorletWavelet(freq, convert(T, k0), convert(T, (4π)/(k0 + sqrt(2 + k0^2))))

# Generate daughter wavelet (samples x frequencies)
function wavebases{T}(w::MorletWavelet{T}, n::Int, fs::Real=1)
    df = 2π * fs / n
    normconst = sqrt(df) / sqrt(sqrt(π) * n)
    k0 = w.k0

    bases = Array(T, div(n, 2)+1, length(w.freq))
    @inbounds begin
        for k = 1:length(w.freq)
            scale = 1/(w.freq[k] * w.fourierfactor)
            norm = sqrt(scale) * normconst
            bases[1, k] = zero(T)
            for j = 2:size(bases, 1)
                bases[j, k] = norm * exp(-abs2(scale * df * (j-1) - k0)*0.5)
            end
        end
    end
    bases
end

fourierfactor{T}(w::MorletWavelet{T}) = w.fourierfactor
# std in the frequency domain, in units of Hz
fstd{T}(w::MorletWavelet{T}) =
    [(f * w.fourierfactor) / (sqrt(2) * 2π) for f in w.freq]
# std in the time domain, in units of time
tstd{T}(w::MorletWavelet{T}) =
    [1 / (sqrt(2) * f * w.fourierfactor) for f in w.freq]

# Generalized Morse wavelet (first family only). See:
# Olhede, S. C., & Walden, A. T. (2002). Generalized Morse wavelets.
# IEEE Transactions on Signal Processing, 50(11), 2661–2670.
# doi:10.1109/TSP.2002.804066
# Lilly, J. M., & Olhede, S. C. (2009). Higher-order properties of
# analytic wavelets. IEEE Transactions on Signal Processing, 57(1),
# 146–160. doi:10.1109/TSP.2008.2007607
#
# freq is the peak frequency in Hz
# β and γ are wavelet parameters
immutable MorseWavelet{T} <: MotherWavelet{T}
    freq::Vector{T}
    β::T
    γ::T
end
MorseWavelet{T<:Real}(freq::Vector{T}, β::Real, γ::Real) =
    MorseWavelet{T}(freq, convert(T, β), convert(T, γ))

# Generate daughter wavelet (samples x frequencies)
function wavebases{T}(w::MorseWavelet{T}, n::Int, fs::Real=1)
    γ = w.γ
    β = w.β
    # Peak frequency
    ωs = (β/γ)^(1/γ)
    dω0 = fs / n * ωs
    # Olhede & Walden p. 2663, multiplied by sqrt(ωs/(2*pi*f)) for the
    # frequency with an additional correction by 1/n for the
    # unnormalized inverse FFT
    r = (2β+1)/γ
    α0 = 2^(r/2)*sqrt(γ*ωs*fs)/(sqrt(gamma(r))*n)

    bases = Array(T, div(n, 2)+1, length(w.freq))
    for k = 1:length(w.freq)
        f = w.freq[k]
        # Change in angular frequency with each point of the DFT
        dω = dω0/f
        α = α0*f^(-0.5)
        for j = 1:size(bases, 1)
            ω = dω*(j-1) 
            bases[j, k] = α * ω^β * exp(-ω^γ)
        end
    end
    bases
end

# Fourier factor based on peak frequency
fourierfactor{T}(w::MorseWavelet{T}) = 2*pi*(w.β/w.γ)^(-1/w.γ)
function fstd{T}(w::MorseWavelet{T})
    γ = w.γ
    β = w.β
    ff = fourierfactor(w)
    σ = sqrt(2^(-2/γ)*(exp(lgamma((2*β+3)/γ)-
                           lgamma((2*β+1)/γ)) -
                       exp(2*(lgamma((2*β+2)/γ)-
                              lgamma((2*β+1)/γ)))))
    [(f * ff * σ) / (2 * pi) for f in w.freq]
end
gammatil(x) = gamma(x)/2^x
function tstd{T}(w::MorseWavelet{T})
    γ = w.γ
    β = w.β
    ff = fourierfactor(w)
    σ = sqrt((β^2*gammatil((2β-1)/γ)+γ^2*
             gammatil((2β+2*γ-1)/γ)-2β*γ*
             gammatil((2β+γ-1)/γ))/gammatil((2β+1)/γ))
    [(σ) / (f * ff) for f in w.freq]
end

#
# Functions for applying wavelets to data
#
immutable ContinuousWaveletTransform{T,S,P1,P2}
    fftin::Vector{T}
    fftout::Vector{S}
    ifftwork::Vector{S}
    bases::Array{T,2}
    coi::Vector{T}
    p1::P1
    p2::P2
end

function ContinuousWaveletTransform{T}(w::MotherWavelet{T}, nfft::Int, fs::Real=1;
                                       coi::Vector=scale!(tstd(w), 2*fs))
    fftin = Array(T, nfft)
    fftout = zeros(Complex{T}, div(nfft, 2)+1)
    ifftwork = zeros(Complex{T}, nfft)
    bases = wavebases(w, nfft, fs)
    length(coi) == size(bases, 2) || isempty(coi) || error("length of coi must match number of frequencies")
    ContinuousWaveletTransform(fftin, fftout, ifftwork, bases, coi, plan_rfft(fftin), plan_bfft!(ifftwork))
end

function evaluate!{T,S<:AbstractFloat}(out::Array{Complex{S}, 2}, t::ContinuousWaveletTransform{T},
                                       signal::AbstractVector{T})
    @inbounds begin
        fftin = t.fftin
        fftout = t.fftout
        ifftwork = t.ifftwork
        bases = t.bases
        coi = t.coi

        nsignal = length(signal)
        nfft = length(fftin)
        nrfft = length(fftout)

        nsignal <= nfft || error("signal exceeds length of transform plan")
        size(out, 1) == length(signal) || error("first dimension of out must match length of signal")
        size(out, 2) == size(bases, 2) || error("second dimension of out must match number of wavelets")

        # Compute signal mean ignoring NaN
        m = 0.0
        for i = 1:nsignal
            v = signal[i]
            m += isnan(v) ? 0.0 : v
        end
        m = convert(T, m/nsignal)
        # Subtract signal mean
        for i = 1:nsignal
            fftin[i] = signal[i] - m
        end
        fftin[nsignal+1:nfft] = zero(T)

        # Interpolate NaN values in input and save their indices to
        # set output to NaN later
        discard_samples = falses(length(signal))
        oldv = zero(eltype(fftin))
        i = 1
        while i <= nsignal
            v = fftin[i]
            if isnan(v)
                nanstart = i

                i += 1
                while i <= nsignal && isnan(fftin[i])
                    i += 1
                end

                nanlen = (i - nanstart)
                st = i > nsignal ? zero(eltype(fftin)) : (fftin[i] - oldv)/(nanlen+1)
                for j = 1:nanlen
                    fftin[nanstart+j-1] = oldv+st*j
                end
                discard_samples[nanstart:i-1] = true
            else
                oldv = v
                i += 1
            end
        end
        # This simplifies replacement with NaN below
        discard_samples[1] = false
        discard_samples[end] = false
        discard_sample_indices = find(discard_samples)

        # Perform FFT of padded signal
        A_mul_B!(fftout, t.p1, fftin)

        for k = 1:size(bases, 2)
            # Multiply by wavelet
            for j = 1:nrfft
                ifftwork[j] = fftout[j] * bases[j, k]
            end

            # We only compute the real FFT, but we may need the imaginary
            # frequencies for some mother wavelets as well
            offset = nrfft + 1 + isodd(nfft)
            for j = nrfft+1:size(bases, 1)
                ifftwork[j] = conj(fftout[offset-j]) * bases[j, k]
            end

            # Zero remaining frequencies
            ifftwork[size(bases, 1)+1:nfft] = zero(Complex{T})

            # Perform FFT
            A_mul_B!(ifftwork, t.p2, ifftwork)

            # Copy to output array
            copy!(out, nsignal*(k-1)+1, ifftwork, 1, nsignal)

            if !isempty(coi)
                # Set NaNs at edges
                coi_length = ceil(Int, coi[k])
                out[1:min(coi_length, nsignal), k] = NaN
                out[max(nsignal-coi_length+1, 1):end, k] = NaN

                # Set NaNs for gaps
                for i in discard_sample_indices
                    out[i, k] = NaN
                    if !discard_samples[i+1]
                        out[i:min(i+coi_length, nsignal), k] = NaN
                    end
                    if !discard_samples[i-1]
                        out[max(i-coi_length, 1):i, k] = NaN
                    end
                end
            end
        end
    end
    out
end

# Friendly interface to ContinuousWaveletTransform
function cwt{T<:Real}(signal::Vector{T}, w::MotherWavelet, fs::Real=1)
    t = ContinuousWaveletTransform(w, nextfastfft(length(signal)), fs)
    evaluate!(Array(Complex{T}, length(signal), size(t.bases, 2)), t, signal)
end
# FrequencyDomainAnalysis.jl
# Tools for spectral density estimation and analysis of phase relationships
# between sets of signals.

# Copyright (C) 2013   Simon Kornblith

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

module FrequencyDomainAnalysis
using NumericExtensions

export PowerSpectrum, PowerSpectrumVariance, CrossSpectrum, Coherence, Coherency, PLV, PPC, PLI,
       PLI2Unbiased, WPLI, WPLI2Debiased, ShiftPredictor, dpss, multitaper, psd, xspec, coherence,
       spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1, pfppc2, pointfieldstat,
       allpairs, frequencies

#
# Statistics computed on transformed data
#
abstract TransformStatistic{T<:Real}
abstract PairwiseTransformStatistic{T<:Real} <: TransformStatistic{T}

# Generate a new PairwiseTransformStatistic, including constructors
macro pairwisestat(name, xtype)
    esc(quote
        type $name{T<:Real} <: PairwiseTransformStatistic{T}
            pairs::Array{Int, 2}
            x::$xtype
            nepochs::Vector{Int}
            tmp::BitVector
            $name() = new()
            $name(pairs::Array{Int, 2}) = new(pairs)
        end
        $name() = $name{Float64}()
    end)
end

# Most PairwiseTransformStatistics will initialize their fields the same way
function init{T}(s::PairwiseTransformStatistic{T}, nout, nchannels, ntapers)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.x = zeros(eltype(fieldtype(s, :x)), nout, size(s.pairs, 2))
    s.nepochs = zeros(Int, size(s.pairs, 2))
    s.tmp = falses(nchannels)
end

# Create accumulate function that loops over pairs of channels, performing some transform for each
macro accumulatebypair(stat, arr, freqindex, pairindex, ch1ft, ch2ft, code)
    quote
        # Split out the two FFTs to allow efficient computation in shift predictor case
        function $(esc(:accumulatepairs))(s::$stat, fftout1, fftout2, itaper)
            $arr = s.x
            tmp = s.tmp
            pairs = s.pairs
            nepochs = s.nepochs
            @inbounds begin
                # Find channels with NaNs
                for i = 1:size(fftout1, 2)
                    good = true
                    for j = 1:size(fftout1, 1)
                        good &= !isnan(real(fftout1[j, i])) & !isnan(real(fftout2[j, i]))
                    end
                    tmp[i] = good
                end

                for $pairindex = 1:size(pairs, 2)
                    ch1 = pairs[1, $pairindex]
                    ch2 = pairs[2, $pairindex]

                    # Skip channels with NaNs
                    if !tmp[ch1] || !tmp[ch2] continue end
                    nepochs[$pairindex] += 1

                    for $freqindex = 1:size(fftout1, 1)
                        $ch1ft = fftout1[$freqindex, ch1]
                        $ch2ft = fftout2[$freqindex, ch2]
                        $code
                    end
                end
            end
        end
    end
end

accumulate(s::PairwiseTransformStatistic, fftout, itaper) =
    accumulatepairs(s, fftout, fftout, itaper)

#
# Power spectrum
#
type PowerSpectrum{T<:Real} <: TransformStatistic{T}
    x::Array{T, 2}
    nepochs::Vector{Int}
    PowerSpectrum() = new()
end
PowerSpectrum() = PowerSpectrum{Float64}()
function init{T}(s::PowerSpectrum{T}, nout, nchannels, ntapers)
    s.x = zeros(T, nout, nchannels)
    s.nepochs = zeros(Int, nchannels)
end
function accumulate(s::PowerSpectrum, fftout, itaper)
    A = s.x
    @inbounds for i = 1:size(fftout, 2)
        # Skip channels with NaNs
        good = true
        for j = 1:size(fftout, 1)
            good &= !isnan(real(fftout[j, i]))
        end
        if !good continue end

        s.nepochs[i] += 1
        for j = 1:size(fftout, 1)
            A[j, i] += abs2(fftout[j, i])
        end
    end
end
finish(s::PowerSpectrum) = scale!(s.x, 1./s.nepochs)

#
# Variance of power spectrum across trials
#
type PowerSpectrumVariance{T<:Real} <: TransformStatistic{T}
    x::Array{T, 3}
    ntrials::Vector{Int}
    ntapers::Int
    PowerSpectrumVariance() = new()
end
PowerSpectrumVariance() = PowerSpectrum{Float64}()
function init{T}(s::PowerSpectrumVariance{T}, nout, nchannels, ntapers)
    s.x = zeros(T, 3, nout, nchannels)
    s.ntrials = zeros(Int, nchannels)
    s.ntapers = ntapers
end
function accumulate(s::PowerSpectrumVariance, fftout, itaper)
    @inbounds begin
        A = s.x
        for i = 1:size(fftout, 2), j = 1:size(fftout, 1)
            A[1, j, i] += abs2(fftout[j, i])
        end
        if itaper == s.ntapers
            for i = 1:size(A, 3)
                # Skip channels with NaNs
                good = true
                for j = 1:size(A, 2)
                    good &= !isnan(real(A[1, j, i]))
                end
                if !good continue end

                # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
                n = (s.ntrials[i] += 1) # n = n + 1
                for j = 1:size(A, 2)
                    x = A[1, j, i]
                    mean = A[2, j, i]
                    delta = x - mean
                    mean = mean + delta/n
                    A[3, j, i] += delta*(x - mean) # M2 = M2 + delta*(x - mean)
                    A[2, j, i] = mean
                end
            end
        end
    end
end
finish(s::PowerSpectrumVariance) = scale!(squeeze(s.x[3, :, :], 1), 1./(s.ntrials - 1))

#
# Cross spectrum
#
@pairwisestat CrossSpectrum Matrix{Complex{T}}
@accumulatebypair CrossSpectrum A j i x y begin
    A[j, i] += conj(x)*y
end
finish(s::CrossSpectrum) = scale!(s.x, 1./s.nepochs)

#
# Coherency and coherence
#
for sym in (:Coherency, :Coherence)
    @eval begin
        type $sym{T<:Real} <: PairwiseTransformStatistic{T}
            pairs::Array{Int, 2}
            psd::PowerSpectrum{T}
            xspec::CrossSpectrum{T}
            $sym() = new()
            $sym(pairs::Array{Int, 2}) = new(pairs)
        end
        $sym() = $sym{Float64}()
    end
end
function init{T}(s::Union(Coherency{T}, Coherence{T}), nout, nchannels, ntapers)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.psd = PowerSpectrum{T}()
    s.xspec = CrossSpectrum{T}(s.pairs)
    init(s.psd, nout, nchannels, ntapers)
    init(s.xspec, nout, nchannels, ntapers)
end
function accumulatepairs(s::Union(Coherency, Coherence), fftout1, fftout2, itaper)
    accumulate(s.psd, fftout1, itaper)
    accumulatepairs(s.xspec, fftout1, fftout2, itaper)
end
function finish(s::Coherency)
    psd = finish(s.psd)
    xspec = finish(s.xspec)
    pairs = s.pairs
    for i = 1:size(pairs, 2)
        ch1 = pairs[1, i]
        ch2 = pairs[2, i]
        for j = 1:size(xspec, 1)
            xspec[j, i] = xspec[j, i]/sqrt(psd[j, ch1]*psd[j, ch2])
        end
    end
    xspec
end
function finish{T}(s::Coherence{T})
    psd = finish(s.psd)
    xspec = finish(s.xspec)
    out = zeros(T, size(xspec, 1), size(xspec, 2))
    pairs = s.pairs
    for i = 1:size(pairs, 2)
        ch1 = pairs[1, i]
        ch2 = pairs[2, i]
        for j = 1:size(xspec, 1)
            out[j, i] = abs(xspec[j, i])/sqrt(psd[j, ch1]*psd[j, ch2])
        end
    end
    out
end

#
# Phase locking value and pairwise phase consistency
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
@pairwisestat PLV Matrix{Complex{T}}
@pairwisestat PPC Matrix{Complex{T}}
@accumulatebypair Union(PLV, PPC) A j i x y begin
    # Add phase difference between pair as a unit vector
    z = conj(x)*y
    A[j, i] += z/abs(z)
    # Faster, but less precise
    #A[j, i] += z*(1/sqrt(abs2(real(z))+abs2(imag(z))))
end
finish(s::PLV) = scale!(abs(s.x), 1./s.nepochs)
function finish{T}(s::PPC{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2)
        nepochs = s.nepochs[i]
        for j = 1:size(s.x, 1)
            # This is equivalent to the formulation in Vinck et al. (2010), since
            # 2*sum(unique pairs) = sum(trials)^2-n. 
            out[j, i] = (abs2(s.x[j, i])-nepochs)/(nepochs*(nepochs-1))
        end
    end
    out
end

#
# Phase lag index and unbiased PLI^2
#
# For PLI, see Stam, C. J., Nolte, G., & Daffertshofer, A. (2007).
# Phase lag index: Assessment of functional connectivity from multi
# channel EEG and MEG with diminished bias from common sources.
# Human Brain Mapping, 28(11), 1178–1193. doi:10.1002/hbm.20346
#
# For unbiased PLI^2, see Vinck, M., Oostenveld, R., van
# Wingerden, M., Battaglia, F., & Pennartz, C. M. A. (2011). An
# improved index of phase-synchronization for electrophysiological data
# in the presence of volume-conduction, noise and sample-size bias.
# NeuroImage, 55(4), 1548–1565. doi:10.1016/j.neuroimage.2011.01.055
@pairwisestat PLI Matrix{Int}
@pairwisestat PLI2Unbiased Matrix{Int}
@accumulatebypair Union(PLI, PLI2Unbiased) A j i x y begin
    z = imag(conj(x)*y)
    if z != 0
        A[j, i] += 2*(z > 0)-1
    end
end
function finish{T}(s::PLI{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2)
        nepochs = s.nepochs[i]
        for j = 1:size(s.x, 1)
            out[j. i] = abs(s.x[j, i])/nepochs
        end
    end
    out
end
function finish{T}(s::PLI2Unbiased{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2)
        nepochs = s.nepochs[i]
        for j = 1:size(s.x, 1)
            out[j, i] = (nepochs * abs2(s.x[j, i]/nepochs) - 1)/(nepochs - 1)
        end
    end
    out
end

#
# Weighted phase lag index
#
# See Vinck et al. (2011) as above.
@pairwisestat WPLI Array{T,3}
# We need 2 fields per freq/channel in s.x
function init{T}(s::WPLI{T}, nout, nchannels, ntapers)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.x = zeros(eltype(fieldtype(s, :x)), 2, nout, size(s.pairs, 2))
    s.nepochs = zeros(Int, size(s.pairs, 2))
    s.tmp = falses(nchannels)
end
@accumulatebypair WPLI A j i x y begin
    z = imag(conj(x)*y)
    A[1, j, i] += z
    A[2, j, i] += abs(z)
end
function finish{T}(s::WPLI{T}, nepochs)
    out = zeros(T, size(s.x, 2), size(s.x, 3))
    for i = 1:size(out, 2), j = 1:size(out, 1)
        out[j, i] = abs(s.x[1, j, i])/s.x[2, j, i]
    end
    out
end

#
# Debiased (i.e. still somewhat biased) WPLI^2
#
# See Vinck et al. (2011) as above.
@pairwisestat WPLI2Debiased Array{T,3}
# We need 3 fields per freq/channel in s.x
function init{T}(s::WPLI2Debiased{T}, nout, nchannels, ntapers)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.x = zeros(eltype(fieldtype(s, :x)), 3, nout, size(s.pairs, 2))
    s.nepochs = zeros(Int, size(s.pairs, 2))
    s.tmp = falses(nchannels)
end
@accumulatebypair WPLI2Debiased A j i x y begin
    z = imag(conj(x)*y)
    A[1, j, i] += z
    A[2, j, i] += abs(z)
    A[3, j, i] += abs2(z)
end
function finish{T}(s::WPLI2Debiased{T})
    out = zeros(T, size(s.x, 2), size(s.x, 3))
    for i = 1:size(out, 2), j = 1:size(out, 1)
        imcsd = s.x[1, j, i]
        absimcsd = s.x[2, j, i]
        sqimcsd = s.x[3, j, i]
        out[j, i] = (abs2(imcsd) - sqimcsd)/(absimcsd - sqimcsd)
    end
    out
end

#
# Shift predictors
#
type ShiftPredictor{T<:Real,S<:PairwiseTransformStatistic} <: TransformStatistic{T}
    stat::S
    first::Array{Complex{T}, 3}
    previous::Array{Complex{T}, 3}
    isfirst::Bool
end
ShiftPredictor{T}(stat::PairwiseTransformStatistic{T}) =
    ShiftPredictor{T,typeof(stat)}(stat, Array(Complex{T}, 0, 0, 0), Array(Complex{T}, 0, 0, 0), true)

function init{T}(s::ShiftPredictor{T}, nout, nchannels, ntapers)
    s.first = Array(Complex{T}, nout, nchannels, ntapers)
    s.previous = Array(Complex{T}, nout, nchannels, ntapers)
    s.isfirst = true
    init(s.stat, nout, nchannels, ntapers)
end
function accumulate(s::ShiftPredictor, fftout, itaper)
    first = pointer_to_array(pointer(s.first, size(s.first, 1)*size(s.first, 2)*(itaper - 1)+1),
                             (size(s.first, 1), size(s.first, 2)), false)
    previous = pointer_to_array(pointer(s.previous, size(s.previous, 1)*size(s.previous, 2)*(itaper - 1)+1),
                                (size(s.previous, 1), size(s.previous, 2)), false)
    if s.isfirst
        copy!(first, fftout)
        s.isfirst = itaper != size(s.first, 3)
    else
        accumulatepairs(s.stat, previous, fftout, itaper)
    end
    copy!(previous, fftout)
end
function finish(s::ShiftPredictor)
    if !s.isfirst
        for itaper = 1:size(s.first, 3)
            first = pointer_to_array(pointer(s.first, size(s.first, 1)*size(s.first, 2)*(itaper - 1)+1),
                                     (size(s.first, 1), size(s.first, 2)), false)
            previous = pointer_to_array(pointer(s.previous, size(s.previous, 1)*size(s.previous, 2)*(itaper - 1)+1),
                                        (size(s.previous, 1), size(s.previous, 2)), false)
            accumulatepairs(s.stat, previous, first, itaper)
        end
    end
    finish(s.stat)
end

#
# Core functionality
#

# Compute discrete prolate spheroid sequences (Slepian tapers)
#
# See Gruenbacher, D. M., & Hummels, D. R. (1994). A simple algorithm
# for generating discrete prolate spheroidal sequences. IEEE
# Transactions on Signal Processing, 42(11), 3276-3278.
function dpss(n::Int, nw::Real, ntapers::Int=iceil(2*nw)-1)
    # Construct symmetric tridiagonal matrix
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

# Compute Hann window for n samples
hann(n) = 0.5*(1-cos(2*pi*(0:n-1)/(n-1)))

# Compute frequencies based on length of padded FFT
function frequencies(nfft::Int, fs::Real=1.0)
    freq = (0:div(nfft, 2))*(fs/nfft)
    (freq, 1:length(freq))
end

# Compute frequencies based on length of padded FFT, with limits
function frequencies(nfft::Int, fs::Real, fmin::Real, fmax::Real=Inf)
    (freq,) = frequencies(nfft, fs)
    freqrange = searchsortedfirst(freq, fmin):(fmax == Inf ? length(freq) : searchsortedlast(freq, fmax))
    (freq[freqrange], freqrange)
end

# Compute frequencies based on data length, assuming padding to next power of 2
frequencies{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                     fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf) =
    frequencies(nextpow2(size(A, 1)), fs, fmin, fmax)

# Perform tapered FFT of continuous signals
# A is samples x channels x trials
function multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                             stats::(TransformStatistic, TransformStatistic...);
                             tapers::Union(Vector, Matrix)=dpss(size(A, 1), 4), nfft::Int=nextpow2(size(A, 1)),
                             fs::Real=1.0, freqrange::Range1{Int}=1:(nfft >> 1 + 1))
    n = size(A, 1)
    nout = nfft >> 1 + 1
    ntapers = size(tapers, 2)
    nchannels = size(A, 2)
    multiplier = sqrt(2/fs)

    for stat in stats
        init(stat, length(freqrange), nchannels, ntapers)
    end

    dtype = outputtype(T)
    fftin = zeros(dtype, nfft, size(A, 2))
    fftout = Array(Complex{dtype}, nout, size(A, 2))
    #fftview = sub(fftout, freqrange, 1:size(A,2))
    fftview = zeros(Complex{dtype}, length(freqrange), size(A, 2))

    p = FFTW.Plan(fftin, fftout, 1, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)

    for j = 1:size(A, 3), i = 1:ntapers
        @inbounds for k = 1:nchannels, l = 1:n
            fftin[l, k] = A[l, k, j]*tapers[l, i]
        end

        FFTW.execute(dtype, p.plan)
        copyscalefft!(fftview, fftout, freqrange, multiplier, nfft)

        for stat in stats
            accumulate(stat, fftview, i)
        end
    end

    [finish(stat) for stat in stats]
end

# Calling with types instead of instances
multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                    stat::(DataType, DataType...); kw...) =
    multitaper(A, tuple([x{outputtype(T)}() for x in stat]...); kw...)

# Calling with a single type or instance
multitaper{T<:Real,S<:TransformStatistic}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                                          stat::Union(S, Type{S}); kw...) = multitaper(A, (stat,); kw...)[1]

# Perform tapered FFT of individual spikes embedded in a continuous signal
function spiketriggeredspectrum{T<:Integer,S<:Real}(points::AbstractVector{T}, field::AbstractVector{S},
                                                    window::Range1{Int};
                                                    nfft::Int=nextpow2(length(window)),
                                                    freqrange::Range1{Int}=1:(nfft >> 1 + 1),
                                                    tapers::Union(Vector, Matrix)=hann(length(window)))#dpss(length(window), length(window)/fs*10))
    n = length(window)
    nfreq = length(freqrange)
    npoints = length(points)
    nfield = length(field)
    dtype = outputtype(S)

    # getindex() on ranges is a function call...
    freqoff = freqrange[1]-1
    winoff = window[1]-1

    fftin = zeros(dtype, nfft)
    sts = Array(Complex{dtype}, length(freqrange), size(tapers, 2), npoints)
    stw = Array(S, n, npoints)
    fftout = zeros(Complex{dtype}, nfft >> 1 + 1)

    p = FFTW.Plan(fftin, fftout, 1, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)

    # Calculate PSD for each point and taper
    @inbounds begin
        for j = 1:npoints
            point = points[j]
            if point-window[1] < 1
                error("Not enough samples before point $j")
            end
            if point+window[end] > nfield
                error("Not enough samples after point $j")
            end

            for l = 1:n
                stw[l, j] = field[point+winoff+l]
            end

            for i = 1:size(tapers, 2)
                for l = 1:n
                    fftin[l] = tapers[l, i]*stw[l, j]
                end
                FFTW.execute(p.plan, fftin, fftout)
                for l = 1:nfreq
                    sts[l, i, j] = fftout[freqoff+l]
                end
            end
        end
    end

    (sts, stw)
end

#
# Spike-field coherence
#
# See Fries, P., Roelfsema, P. R., Engel, A. K., König, P., & Singer,
# W. (1997). Synchronization of oscillatory responses in visual cortex
# correlates with perception in interocular rivalry. Proceedings of the
# National Academy of Sciences, 94(23), 12699–12704.
#
# For bias correction, see Grasse, D. W., & Moxon, K. A. (2010).
# Correcting the bias of spike field coherence estimators due to a
# finite number of spikes. Journal of Neurophysiology, 104(1), 548–558.
# doi:10.1152/jn.00610.2009
#
# This is not (quite) the same thing as the spike field coherence
# measure in Jarvis & Mitra, 2001 and implemented in Chronux!
function pfcoherence{T<:Real,S<:Real}(sts::Array{Complex{T},3}, stw::Array{S,2};
                                      nfft::Int=nextpow2(length(sta)),
                                      freqrange::Range1{Int}=1:(nfft >> 1 + 1),
                                      tapers::Union(Vector, Matrix)=hann(length(window)),
                                      debias::Bool=false)
    if size(sts, 1) != length(freqrange)
        error("Size of spike triggered spectrum must match number of frequencies to use")
    end
    npoints = size(sts, 3)

    # Calculate PSD of spike triggered average for each taper
    fftin = zeros(T, nfft, size(tapers, 2))
    sta = mean(stw, 2)
    @inbounds begin
        for i = 1:size(tapers, 2), l = 1:size(stw, 1)
            fftin[l, i] = tapers[l, i]*sta[l]
        end
    end
    meanpsd = abs2(rfft(fftin, 1)[freqrange, :])
    psdsum = sqsum!(Array(T, size(sts, 1), size(sts, 2)), sts, 3)

    # Calculate spike field coherence
    sfc = zeros(T, size(sts, 1))
    @inbounds begin
        for i = 1:size(psdsum, 2), l = 1:size(psdsum, 1)
            c = npoints*meanpsd[l, i]/psdsum[l, i]
            if debias
                c = (npoints*c - 1)/(npoints - 1)
            end
            sfc[l] += c
        end
    end
    scale!(sfc, 1.0/size(tapers, 2))
end

#
# Point-field PLV/PPC
#
# See Vinck, M., Battaglia, F. P., Womelsdorf, T., & Pennartz, C.
# (2012). Improved measures of phase-coupling between spikes and the
# Local Field Potential. Journal of Computational Neuroscience, 33(1),
# 53–75. doi:10.1007/s10827-011-0374-4
#
# The code below computes these statistics using more efficient methods
# than those described in Vinck et al. (2012). These efficient methods
# are also used in FieldTrip (https://github.com/fieldtrip/fieldtrip),
# although there is a bug in FieldTrip that makes its comptutations
# incorrect. Derivation is available upon request.

# Computes the sum of complex phases over third dimension
function phasesum{T<:Real}(sts::Array{Complex{T},3})
    ps = zeros(Complex{T}, size(sts, 1), size(sts, 2))
    for i = 1:size(sts, 3), j = 1:size(sts, 2), k = 1:size(sts, 1)
        v = sts[k, j, i]
        ps[j, k] += v/abs(v)
    end
end

# Computes the sum of phases for each trial
function phasesumtrial{T<:Real, U<:Integer}(sts::Array{Complex{T}, 3}, trials::AbstractVector{U})
    if size(sts, 3) != length(trials)
        error("length of trials does not match third dimension of spike-triggered spectrum")
    end
    nutrial = length(unique(trials))
    pstrial = zeros(Complex{T}, size(sts, 1), size(sts, 2), nutrial)
    nintrial = zeros(Int, nutrial)

    trial = 1
    lasttrial = trials[1]
    @inbounds begin
        for i = 1:size(sts, 3)
            trial += trials[i] != lasttrial
            lasttrial = trials[i]

            nintrial[trial] += 1
            for j = 1:size(sts, 2), k = 1:size(sts, 1)
                v = sts[k, j, i]
                v /= abs(v)
                pstrial[k, j, trial] += v
            end
        end
    end

    (pstrial, nintrial)
end

# Computes the difference between the squared sum of pstrial along the
# third dimension and the sum of squares of pstrial along the third
# dimension. If jackknife is true, then also computes the jackknifed
# difference.
function pfdiffsq{T}(pstrial::Array{T,3}, jackknife::Bool)
    pssum = sum(pstrial, 3)
    pssqsum = sqsum(pstrial, 3)
    tval = vec(sum(abs2(pssum) - pssqsum, 2))

    if jackknife
        jnval = zeros(T, size(pstrial, 1), size(pstrial, 3))
        @inbounds begin
            for i = 1:size(pstrial, 3), j = 1:size(pstrial, 2), k = 1:size(pstrial, 1)
                v = pstrial[k, j, i]
                jnval[k, i] += abs2(pssum[k, j] - v) - (pssqsum[k, j] - abs2(v))
            end
        end
    else
        jnval = zeros(T, 0, 0)
    end
    (tval, jnval)
end

jnvar(jnval, dim) = var(jnval, dim)*(size(jnval, dim) - 1)/sqrt(size(jnval, dim))

# Computes the PLV
pfplv{T<:Real}(sts::Array{Complex{T},3}) =
    amean(scale!(phasesum(sts), 1.0/size(sts, 3)), 2)

# Computes the PPC0 statistic from Vinck et al. (2012), which assumes
# that the LFPs surrounding all spikes are independent
pfppc0{T<:Real}(sts::Array{Complex{T}, 3}) =
    (abs2(phasesum(sts)) - size(sts, 3))/(size(sts, 3)*(size(sts, 3) - 1))

# Computes the PPC1 statistic from Vinck et al. (2012), which accounts
# for statistical dependence of spike-LFP phases within a trial
function pfppc1{T<:Real, U<:Integer}(sts::Array{Complex{T}, 3}, trials::AbstractVector{U};
                                     estimatevar::Bool=false)
    pstrial, nintrial = phasesumtrial(sts, trials)
    tval, jnval = pfdiffsq(pstrial, estimatevar)

    nintrial2sum = sqsum(nintrial, estimatevar)
    scale!(tval, 1.0/(size(pmtrial, 2) * (abs2(size(sts, 3)) - nintrial2sum)))

    if estimatevar
        scale!(jnval, 1.0/(size(pmtrial, 2) * (abs2(size(sts, 3) - nintrial) -
                                               (nintrial2sum - abs2(nintrial)))))
        (tval, jnvar(jnval, 2))
    else
        tval
    end
end

# Computes the PPC2 statistic from Vinck et al. (2012), which accounts
# for statistical dependence of spike-LFP phases within a trial as well
# as dependence between spike count and phase
function pfppc2{T<:Real, U<:Integer}(sts::Array{Complex{T}, 3}, trials::AbstractVector{U};
                                     estimatevar::Bool=false)
    pmtrial, nintrial = phasesumtrial(sts, trials)
    @inbounds begin
        for i = 1:size(pmtrial, 3)
            @assert nintrial[i] != 0
            m = 1.0/nintrial[i]
            for j = 1:size(pmtrial, 2), k = 1:size(pmtrial, 1)
                pmtrial[k, j, i] *= m
            end
        end
    end

    tval, jnval = pfdiffsq(pmtrial, estimatevar)
    scale!(tval, 1.0/(size(pmtrial, 2) * size(pmtrial, 3) * (size(pmtrial, 3) - 1)))

    if estimatevar
        scale!(jnval, 1.0/(size(pmtrial, 2) * (size(pmtrial, 3) - 1) * (size(pmtrial, 3) - 2)))
        (tval, jnvar(jnval, 2))
    else
        tval
    end
end

#
# Convenience functions
#

psd{T<:Real}(A::AbstractArray{T}; kw...) =
    multitaper(A, (PowerSpectrum{T}(),); kw...)[1]
xspec{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (CrossSpectrum{T}(),); kw...)[1]
coherence{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (Coherence{T}(),); kw...)[1]

#
# Helper functions
#

# Get all pairs of channels
function allpairs(n)
    pairs = Array(Int, 2, binomial(n, 2))
    k = 0
    for i = 1:n-1, j = i+1:n
        k += 1
        pairs[1, k] = i
        pairs[2, k] = j
    end
    pairs
end

# Copy FFT from one buffer to another, including only frequencies in
# frange, simultaneously multiplying by multiplier, and correcting
# edges according to nfft
function copyscalefft!(fftview, fftout, frange, multiplier, nfft)
    divisor = 1/sqrt(2)
    @inbounds for k = 1:size(fftout, 2)
        for l = 1:length(frange)
            fftview[l, k] = fftout[frange[l], k]*multiplier
        end

        # Correct edges of FFT so that all power is in the same units
        # This is necessary because the FFT divides power associated
        # with real signals between positive and negative frequencies,
        # except for power at 0 and FMAX. We don't compute the
        # negative frequencies, but we do need to scale the FFT so
        # that the power is correct.
        if frange[1] == 1
            fftview[1, k] *= divisor
        end
        if frange[end] == size(fftout, 1) && iseven(nfft)
            fftview[end, k] *= divisor
        end
    end
end

# Get the equivalent complex type for a given type
complextype{T<:Complex}(::Type{T}) = T
complextype{T}(::Type{T}) = Complex{T}

# Get preferred output type for a given input type
outputtype{T<:FloatingPoint}(::Type{T}) = T
outputtype(::Union(Type{Int8}, Type{Uint8}, Type{Int16}, Type{Uint16})) = Float64
outputtype{T<:Real}(::Type{T}) = Float64
end
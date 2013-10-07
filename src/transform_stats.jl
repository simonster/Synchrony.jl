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

export PowerSpectrum, PowerSpectrumVariance, CrossSpectrum, Coherence, Coherency, PLV, PPC, PLI,
       PLI2Unbiased, WPLI, WPLI2Debiased, ShiftPredictor, Jackknife, allpairs, applystat

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
            n::Matrix{Int32}
            $name() = new()
            $name(pairs::Array{Int, 2}) = new(pairs)
        end
        $name() = $name{Float64}()
    end)
end

# Most PairwiseTransformStatistics will initialize their fields the same way
function init{T}(s::PairwiseTransformStatistic{T}, nout, nchannels, ntapers, ntrials)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.x = zeros(eltype(fieldtype(s, :x)), datasize(s, nout))
    s.n = zeros(Int32, nout, size(s.pairs, 2))
end
datasize(s::PairwiseTransformStatistic, nout) = (nout, size(s.pairs, 2))

# Create accumulate function that loops over pairs of channels,
# performing some transform for each
macro accumulatebypair(stat, arr, freqindex, pairindex, ch1ft, ch2ft, code)
    quote
        # fftout1 and fftout2 are split out above to allow efficient
        # computation of the shift predictor
        # s.x is split out to allow efficient jackknifing
        function $(esc(:accumulateinternal))($arr, n, s::$stat, fftout1, fftout2, itaper)
            pairs = s.pairs
            @inbounds begin
                for $pairindex = 1:size(pairs, 2)
                    ch1 = pairs[1, $pairindex]
                    ch2 = pairs[2, $pairindex]

                    for $freqindex = 1:size(fftout1, 1)
                        $ch1ft = fftout1[$freqindex, ch1]
                        if isnan(real($ch1ft)) continue end
                        $ch2ft = fftout2[$freqindex, ch2]
                        if isnan(real($ch2ft)) continue end

                        n[$freqindex, $pairindex] += 1
                        $code
                    end
                end
            end
        end
    end
end

accumulateinto(x, n, s::PairwiseTransformStatistic, fftout, itaper) =
    accumulateinternal(x, n, s, fftout, fftout, itaper)
accumulatepairs(s::PairwiseTransformStatistic, fftout1, fftout2, itaper) =
    accumulateinternal(s.x, s.n, s, fftout1, fftout2, itaper)
accumulate(s::PairwiseTransformStatistic, fftout, itaper) =
    accumulatepairs(s, fftout, fftout, itaper)

#
# Power spectrum
#
type PowerSpectrum{T<:Real} <: TransformStatistic{T}
    x::Array{T, 2}
    n::Matrix{Int32}
    PowerSpectrum() = new()
end
PowerSpectrum() = PowerSpectrum{Float64}()
function init{T}(s::PowerSpectrum{T}, nout, nchannels, ntapers, ntrials)
    s.x = zeros(T, nout, nchannels)
    s.n = zeros(Int32, nout, nchannels)
end
function accumulate(s::PowerSpectrum, fftout, itaper)
    A = s.x
    n = s.n
    @inbounds for i = 1:size(fftout, 2)
        for j = 1:size(fftout, 1)
            ft = fftout[j, i]
            if isnan(real(ft)) continue end
            n[j, i] += 1
            A[j, i] += abs2(ft)
        end
    end
end
finish(s::PowerSpectrum) = s.x./s.n

#
# Variance of power spectrum across trials
#
type PowerSpectrumVariance{T<:Real} <: TransformStatistic{T}
    x::Array{T, 3}
    trialn::Matrix{Int32}
    ntrials::Matrix{Int32}
    ntapers::Int
    PowerSpectrumVariance() = new()
end
PowerSpectrumVariance() = PowerSpectrumVariance{Float64}()
function init{T}(s::PowerSpectrumVariance{T}, nout, nchannels, ntapers, ntrials)
    s.x = zeros(T, 3, nout, nchannels)
    s.trialn = zeros(Int32, nout, nchannels)
    s.ntrials = zeros(Int32, nout, nchannels)
    s.ntapers = ntapers
end
function accumulate{T}(s::PowerSpectrumVariance{T}, fftout, itaper)
    @inbounds begin
        A = s.x
        trialn = s.trialn

        for i = 1:size(fftout, 2), j = 1:size(fftout, 1)
            ft = fftout[j, i]
            if isnan(real(ft)) continue end
            A[1, j, i] += abs2(ft)
            trialn[j, i] += 1
        end

        if itaper == s.ntapers
            ntrials = s.ntrials
            for i = 1:size(A, 3)
                for j = 1:size(A, 2)
                    # http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
                    if trialn[j, i] == 0; continue; end

                    x = A[1, j, i]/trialn[j, i]
                    A[1, j, i] = zero(T)
                    trialn[j, i] = zero(Int32)

                    n = (ntrials[j, i] += 1) # n = n + 1
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
finish(s::PowerSpectrumVariance) = squeeze(s.x[3, :, :], 1)./(s.ntrials - 1)

#
# Cross spectrum
#
@pairwisestat CrossSpectrum Matrix{Complex{T}}
@accumulatebypair CrossSpectrum A j i x y begin
    A[j, i] += conj(x)*y
end
finish(s::CrossSpectrum) = s.x./s.n

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
function init{T}(s::Union(Coherency{T}, Coherence{T}), nout, nchannels, ntapers, ntrials)
    if !isdefined(s, :pairs); s.pairs = allpairs(nchannels); end
    s.psd = PowerSpectrum{T}()
    s.xspec = CrossSpectrum{T}(s.pairs)
    init(s.psd, nout, nchannels, ntapers, ntrials)
    init(s.xspec, nout, nchannels, ntapers, ntrials)
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
finish(s::PLV) = abs(s.x)./s.n
function finish{T}(s::PPC{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2)
        for j = 1:size(s.x, 1)
            # This is equivalent to the formulation in Vinck et al. (2010), since
            # 2*sum(unique pairs) = sum(trials)^2-n. 
            n = s.n[j, i]
            out[j, i] = (abs2(s.x[j, i])-n)/(n*(n-1))
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
        for j = 1:size(s.x, 1)
            n = s.n[j, i]
            out[j. i] = abs(s.x[j, i])/n
        end
    end
    out
end
function finish{T}(s::PLI2Unbiased{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2)
        for j = 1:size(s.x, 1)
            n = s.n[j, i]
            out[j, i] = (n * abs2(s.x[j, i]/n) - 1)/(n - 1)
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
datasize(s::WPLI, nout) = (2, nout, size(s.pairs, 2))
@accumulatebypair WPLI A j i x y begin
    z = imag(conj(x)*y)
    A[1, j, i] += z
    A[2, j, i] += abs(z)
end
function finish{T}(s::WPLI{T}, n)
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
datasize(s::WPLI, nout) = (3, nout, size(s.pairs, 2))
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
        out[j, i] = (abs2(imcsd) - sqimcsd)/(abs2(absimcsd) - sqimcsd)
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
ShiftPredictor{T}(s::PairwiseTransformStatistic{T}) =
    ShiftPredictor{T,typeof(s)}(s, Array(Complex{T}, 0, 0, 0), Array(Complex{T}, 0, 0, 0), true)

function init{T}(s::ShiftPredictor{T}, nout, nchannels, ntapers, ntrials)
    if ntrials == 1
        error("Need >1 trial to generate shift predictor")
    end
    s.first = Array(Complex{T}, nout, nchannels, ntapers)
    s.previous = Array(Complex{T}, nout, nchannels, ntapers)
    s.isfirst = true
    init(s.stat, nout, nchannels, ntapers, ntrials)
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
# Jackknife
#
type Jackknife{T<:Real,S<:PairwiseTransformStatistic,U<:Number,N} <: TransformStatistic{T}
    stat::S
    count::Int
    xoffset::Int
    x::Array{U,N}
    noffset::Int
    n::Array{Int32,3}

    Jackknife(stat::S) = new(stat)
end
function Jackknife{T}(s::PairwiseTransformStatistic{T})
    dtype = fieldtype(s, :x)
    Jackknife{T,typeof(s),eltype(dtype),ndims(dtype)+1}(s)
end

function init{T,S,U}(s::Jackknife{T,S,U}, nout, nchannels, ntapers, ntrials)
    init(s.stat, nout, nchannels, ntapers, ntrials-1)
    s.count = 0
    xsize = size(s.stat.x)
    s.xoffset = prod(xsize)
    s.x = zeros(U, xsize..., ntrials)
    nsize = size(s.stat.n)
    s.noffset = prod(nsize)
    s.n = zeros(Int32, nsize..., ntrials)
end
function accumulate(s::Jackknife, fftout, itaper)
    # Accumulate into column for trial
    i = (s.count += (itaper == 1))
    x = pointer_to_array(pointer(s.x, s.xoffset*(i-1)+1), size(s.stat.x))
    n = pointer_to_array(pointer(s.n, s.noffset*(i-1)+1), size(s.stat.n))
    accumulateinto(x, n, s.stat, fftout, itaper)
end
function finish{T,S}(s::Jackknife{T,S})
    stat = s.stat
    xsize = size(stat.x)
    nsize = size(stat.n)
    x = s.x
    n = s.n

    xsum = sum(x, ndims(x))

    # Subtract each x and n from the sum
    broadcast!(-, x, xsum, x)
    nsum = sum(n, 3)
    nsub = nsum .- n
    nz = reshape(sum(n .!= 0, 3), size(n, 1), size(n, 2))

    # Compute for first surrogate
    stat.x = pointer_to_array(pointer(x, 1), xsize, false)
    stat.n = pointer_to_array(pointer(nsub, 1), nsize, false)
    m = finish(stat)
    m[n[:, :, 1] .== 0] = 0
    allout = zeros(eltype(m), size(m, 1), size(m, 2), s.count)
    allout[:, :, 1] = m

    # Compute statistic and mean for subsequent surrogates
    for i = 2:s.count
        stat.x = pointer_to_array(pointer(x, s.xoffset*(i-1)+1), xsize, false)
        stat.n = pointer_to_array(pointer(nsub, s.noffset*(i-1)+1), nsize, false)
        out = finish(stat)
        allout[:, :, i] = out
        for j = 1:size(out, 2), k = 1:size(out, 1)
            # Ignore if no tapers
            if n[k, j, i] != 0
                m[k, j] += out[k, j]
            end
        end
    end

    # Divide mean by number of non-zero pairs
    broadcast!(/, m, m, nz)

    # Compute true statistic and bias
    stat.x = squeeze(xsum, ndims(xsum))
    stat.n = squeeze(nsum, 2)
    truestat = finish(stat)
    bias = (nz - 1).*(m - truestat)

    # Compute variance
    variance = zeros(eltype(m), size(m))
    for i = 1:size(allout, 3), j = 1:size(allout, 2), k = 1:size(allout, 1)
        if n[k, j, i] != 0
            variance[k, j] += abs2(allout[k, j, i] - m[k, j])
        end
    end
    broadcast!(/, variance, variance, nz-1)

    (truestat, variance, bias)
end

#
# Apply transform statistic to transformed data
#
# Data is
# frequencies x channels x trials or
# frequencies x channels x ntapers x trials or
# frequencies x time x channels x ntapers x trials
function applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},4})
    init(s, size(data, 1), size(data, 2), size(data, 3), size(data, 4))
    offset = size(data, 1)*size(data, 2)
    for j = 1:size(data, 4), itaper = 1:size(data, 3)
            accumulate(s, pointer_to_array(pointer(data,
                                                   offset*((itaper-1)+size(data, 3)*(j-1))+1),
                                           (size(data, 1), size(data, 2))), itaper)
    end
    finish(s)
end
applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},3}) =
    applystat(s, reshape(data, size(data, 1), size(data, 2), 1, size(data, 3)))
function applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},5})
    out = applystat(s, reshape(data, size(data, 1)*size(data, 2), size(data, 3),
                               size(data, 4), size(data, 5)));
    reshape(out, size(data, 1), size(data, 2), size(out, 2))
end
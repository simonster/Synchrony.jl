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
       PLI2Unbiased, WPLI, WPLI2Debiased, ShiftPredictor, allpairs

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
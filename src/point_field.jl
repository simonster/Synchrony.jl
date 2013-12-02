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

export spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1, pfppc2, pxcorr

#
# Perform tapered FFT of individual spikes embedded in a continuous signal
#
function spiketriggeredspectrum{T<:Integer,S<:Real}(points::AbstractVector{T}, field::AbstractVector{S},
                                                    window::Range1{Int};
                                                    nfft::Int=nextpow2(length(window)),
                                                    freqrange::Range1{Int}=1:(nfft >> 1 + 1),
                                                    tapers::Union(Vector, Matrix)=hanning(length(window)))#dpss(length(window), length(window)/fs*10))
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
                                      tapers::Union(Vector, Matrix)=hanning(length(window)),
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

# Binned cross-correlation of event times
# x and y are monotonically increasing vectors of event times, bins is
# a monotonically increasing vector that specifies time bins for the
# cross-correlation function
function pxcorr(x::AbstractVector, y::AbstractVector, bins::AbstractVector)
    counts = zeros(Int, length(bins))
    nx = length(x)
    minbin = minimum(bins)
    maxbin = maximum(bins)
    nbins = length(bins)
    for yspk in y
        xlo = searchsortedfirst(x, yspk+minbin, Base.Sort.Forward)
        xhi = searchsortedlast(x, yspk+maxbin, xlo, nx, Base.Sort.Forward)
        xlo = xlo < 1 ? 1 : xlo
        xhi = xhi > nx ? nx : xhi
        for i = xlo:xhi
            yspkdiff = x[i] - yspk
            bin = searchsortedfirst(bins, yspkdiff)
            bin = bin > nbins ? nbins : bin
            counts[bin] += 1
        end
    end
    counts
end
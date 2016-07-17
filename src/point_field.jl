export spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1, pfppc2, pxcorr

#
# Perform tapered FFT of individual spikes embedded in a continuous signal
#
function spiketriggeredspectrum{T<:Integer,S<:Real}(points::AbstractVector{T}, field::AbstractVector{S},
                                                    window::UnitRange{Int};
                                                    nfft::Int=nextpow2(length(window)),
                                                    freqrange::UnitRange{Int}=1:(nfft >> 1 + 1),
                                                    tapers::Union{AbstractVector, AbstractMatrix}=hanning(length(window)))
    n = length(window)
    nfreq = length(freqrange)
    npoints = length(points)
    nfield = length(field)
    dtype = outputtype(S)

    freqoff = freqrange[1]-1
    winoff = window[1]-1

    fftin = zeros(dtype, nfft)
    sts = Array(Complex{dtype}, length(freqrange), size(tapers, 2), npoints)
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

            for i = 1:size(tapers, 2)
                for l = 1:n
                    fftin[l] = tapers[l, i]*field[point+winoff+l]
                end
                FFTW.execute(p.plan, fftin, fftout)
                for l = 1:nfreq
                    sts[l, i, j] = fftout[freqoff+l]
                end
            end
        end
    end

    sts
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
function pfcoherence{T<:Real}(sts::Array{Complex{T},3}; debias::Bool=false)
    npoints = size(sts, 3)

    # Calculate PSD of spike triggered average for each taper
    meanpsd = sum(sts, 3)
    psdsum = Base.sumabs2!(sts, 3)

    # Calculate spike field coherence
    sfc = zeros(T, size(sts, 1))
    @inbounds begin
        for i = 1:size(psdsum, 2), l = 1:size(psdsum, 1)
            c = abs2(meanpsd[l, i])/(npoints*psdsum[l, i])
            if debias
                c = (npoints*c - 1)/(npoints - 1)
            end
            sfc[l] += c
        end
    end
    scale!(sfc, 1.0/size(sts, 2))
end

# Jackknife surrogates of point-field coherence
function pfcoherence_jn{T<:Real}(sts::Array{Complex{T},3}; debias::Bool=false)
    npoints = size(sts, 3)

    # Calculate PSD of spike triggered average for each taper
    meanpsd = sum(sts, 3)
    psdsum = Base.sumabs2!(sts, 3)

    # Calculate spike field coherence
    jn = zeros(T, size(sts, 1), size(sts, 3))
    for ispike = 1:1:size(sts, 3), itaper = 1:size(sts, 2), ifreq = 1:size(sts, 1)
        v = sts[ifreq, itaper, ispike]
        c = abs2(meanpsd[ifreq, itaper] - v)/((npoints-1)*(psdsum[ifreq, itaper] - abs2(v)))
        if debias
            c = ((npoints-1)*c - 1)/(npoints - 2)
        end
        jn[ifreq, ispike] += c
    end
    scale!(jn, 1.0/size(sts, 2))
end

#
# Point-field PLV/PPC
#
# See Vinck, M., Battaglia, F. P., Womelsdorf, T., & Pennartz, C.
# (2012). Improved measures of phase-coupling between spikes and the
# Local Field Potential. Journal of Computational Neuroscience, 33(1),
# 53–75. doi:10.1007/s10827-011-0374-4

# Computes the sum of complex phases over third dimension
function phasesum{T<:Real}(sts::Array{Complex{T},3})
    ps = zeros(Complex{T}, size(sts, 1), size(sts, 2))
    for i = 1:size(sts, 3), j = 1:size(sts, 2), k = 1:size(sts, 1)
        v = sts[k, j, i]
        ps[j, k] += v/abs(v)
    end
end

# Computes the sum of phases for each trial
function phasesumtrial{T<:Real, U<:Integer}(sts::AbstractArray{Complex{T}, 3}, trials::AbstractVector{U})
    if size(sts, 3) != length(trials)
        error("length of trials does not match third dimension of spike-triggered spectrum")
    end
    nutrial = length(unique(trials))
    pstrial = similar(sts, Complex{T}, size(sts, 1), size(sts, 2), nutrial)
    fill!(pstrial, zero(Complex{T}))
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

# Compute pfdiffsq with preallocated memory for out, pssum, and
# pssqsum. out is assumed to be zeroed; pssum and pssqsum are not.
function _pfdiffsq!{T}(out::Union{Vector{T}, Matrix{T}}, pssum::Array{Complex{T},2}, pssqsum::Array{T,2},
                       pstrial::AbstractArray{Complex{T},3}, trials=1:size(pstrial, 3), idx::Int=1)
    fill!(pssum, zero(Complex{T}))
    fill!(pssqsum, zero(T))
    @inbounds for itrial = trials, itaper = 1:size(pstrial, 2), ifreq = 1:size(pstrial, 1)
        x = pstrial[ifreq, itaper, itrial]
        pssum[ifreq, itaper] += x
        pssqsum[ifreq, itaper] += abs2(x)
    end
    @inbounds for itaper = 1:size(pstrial, 2), ifreq = 1:size(pstrial, 1)
        out[ifreq, idx] += abs2(pssum[ifreq, itaper]) - pssqsum[ifreq, itaper]
    end
    out
end

# Computes the difference between the squared sum of pstrial along the
# third dimension and the sum of squares of pstrial along the third
# dimension. If "true" is provided as the second argument, perform
# jackknifing. If a number is provided, perform bootstrapping.
function pfdiffsq{T<:Real}(pstrial::AbstractArray{Complex{T},3}, jackknife::Bool)
    pssum = Array(Complex{T}, size(pstrial, 1), size(pstrial, 2))
    pssqsum = Array(T, size(pstrial, 1), size(pstrial, 2))
    tval = _pfdiffsq!(zeros(T, size(pstrial, 1)), pssum, pssqsum, pstrial)

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

function pfdiffsq{T<:Real}(pstrial::AbstractArray{Complex{T},3}, nbootstraps::Int)
    pssum = Array(Complex{T}, size(pstrial, 1), size(pstrial, 2))
    pssqsum = Array(T, size(pstrial, 1), size(pstrial, 2))
    tval = _pfdiffsq!(zeros(T, size(pstrial, 1)), pssum, pssqsum, pstrial)

    bsval = similar(pstrial, T, size(pstrial, 1), nbootstraps)
    fill!(bsval, zero(T))
    @inbounds begin
        trials = Array(Int, size(pstrial, 3))
        for iboot = 1:nbootstraps
            rand!(1:size(pstrial, 3), trials)
            _pfdiffsq!(bsval, pssum, pssqsum, pstrial, trials, iboot)
        end
        retbsval = bsval
    end
    (tval, retbsval)
end

jnvar(jnval, dim) = scale!(var(jnval, dim), (size(jnval, dim) - 1)/size(jnval, dim))

# Computes the PLV
pfplv{T<:Real}(sts::AbstractArray{Complex{T},3}) =
    amean(scale!(phasesum(sts), 1.0/size(sts, 3)), 2)

# Computes the PPC0 statistic from Vinck et al. (2012), which assumes
# that the LFPs surrounding all spikes are independent
pfppc0{T<:Real}(sts::AbstractArray{Complex{T}, 3}) =
    (abs2(phasesum(sts)) - size(sts, 3))/(size(sts, 3)*(size(sts, 3) - 1))

# Computes the PPC1 statistic from Vinck et al. (2012), which accounts
# for statistical dependence of spike-LFP phases within a trial
function pfppc1{T<:Real, U<:Integer}(sts::AbstractArray{Complex{T}, 3}, trials::AbstractVector{U};
                                     estimatevar::Bool=false)
    pstrial, nintrial = phasesumtrial(sts, trials)
    tval, jnval = pfdiffsq(pstrial, estimatevar)

    nintrial2sum = sqsum(nintrial)
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
function pfppc2{T<:Real, U<:Integer}(sts::AbstractArray{Complex{T}, 3}, trials::AbstractVector{U};
                                     estimatevar::Bool=false, nbootstraps::Int=0)
    estimatevar == false || nbootstraps == 0 ||
        error("cannot both bootstrap and estimate variance by jackknifing")

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

    tval, jnval = pfdiffsq(pmtrial, nbootstraps != 0 ? nbootstraps : estimatevar)
    normf = 1.0/(size(pmtrial, 2) * size(pmtrial, 3) * (size(pmtrial, 3) - 1))
    scale!(tval, normf)

    if nbootstraps != 0
        (tval, scale!(jnval, normf))
    elseif estimatevar
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
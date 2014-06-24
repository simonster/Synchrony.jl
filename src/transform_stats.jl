export PowerSpectrum, PowerSpectrumVariance, CrossSpectrum, Coherence, Coherency, PLV, PPC, PLI,
       PLI2Unbiased, WPLI, WPLI2Debiased, JMCircularCorrelation, JCircularCorrelation,
       HurtadoModulationIndex, ShiftPredictor, Jackknife, allpairs, allorderedpairs, applystat,
       permstat, jackknife_bias_var

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

# Get all pairs of channels when order is meaningful
function allorderedpairs(n)
    pairs = Array(Int, 2, abs2(n))
    k = 0
    for i = 1:n, j = 1:n
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
macro pairwisestat(name, xtype, fields...)
    esc(quote
        type $name{T<:Real} <: PairwiseTransformStatistic{T}
            pairs::Array{Int, 2}
            x::$xtype
            $(isempty(fields) ? nothing : fields[1])
            n::Matrix{Int32}
            $name() = new()
            $name(pairs::Array{Int, 2}) = new(pairs)
        end
        $name() = $name{Float64}()
    end)
end
pairs(s::PairwiseTransformStatistic) = x.pairs

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
            sz = size(fftout1, 1)
            @inbounds begin
                for $pairindex = 1:size(pairs, 2)
                    ch1offset = (pairs[1, $pairindex]-1)*sz
                    ch2offset = (pairs[2, $pairindex]-1)*sz
                    pairoffset = ($pairindex-1)*sz

                    for $freqindex = 1:sz
                        $ch1ft = fftout1[$freqindex+ch1offset]
                        $ch2ft = fftout2[$freqindex+ch2offset]
                        if isnan(real($ch1ft)) || isnan(real($ch2ft)) continue end

                        n[$freqindex+pairoffset] += 1
                        $code
                    end
                end
            end
        end
    end
end

function accumulateinto!(x, n, s::PairwiseTransformStatistic, fftout, itaper)
    accumulateinternal(x, n, s, fftout, fftout, itaper)
    true
end
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
accumulate(s::PowerSpectrum, fftout, itaper) =
    accumulateinto!(s.x, s.n, s, fftout, itaper)
function accumulateinto!(A, n, s::PowerSpectrum, fftout, itaper)
    @inbounds for i = 1:size(fftout, 2)
        for j = 1:size(fftout, 1)
            ft = fftout[j, i]
            if isnan(real(ft)) continue end
            n[j, i] += 1
            A[j, i] += abs2(ft)
        end
    end
    true
end
finish{T}(s::PowerSpectrum{T}) = (s.x./s.n)::Array{T,2}

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
finish(s::PowerSpectrumVariance) = squeeze(s.x[3, :, :], 1)./(s.ntrials .- 1)

#
# Cross spectrum
#
@pairwisestat CrossSpectrum Matrix{Complex{T}}
@accumulatebypair CrossSpectrum A j i x y begin
    A[j, i] += conj(x)*y
end
finish{T}(s::CrossSpectrum{T}) = (s.x./s.n)::Array{Complex{T},2}

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

# Compute coherency statistic, 
function finish_coherency!(pairs, psd, xspec)
    Base.sqrt!(psd)
    for k = 1:size(xspec, 3), j = 1:size(pairs, 2)
        ch1 = pairs[1, j]
        ch2 = pairs[2, j]
        for i = 1:size(xspec, 1)
            xspec[i, j, k] = xspec[i, j, k]/(psd[i, ch1, k]*psd[i, ch2, k])
        end
    end
    xspec
end
finish(s::Coherency) = finish_coherency!(s.pairs, finish(s.psd), finish(s.xspec))
finish(s::Coherence) = abs(finish_coherency!(s.pairs, finish(s.psd), finish(s.xspec)))

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
    for i = 1:size(s.x, 2), j = 1:size(s.x, 1)
        n = s.n[j, i]
        out[j, i] = abs(s.x[j, i])/n
    end
    out
end
function finish{T}(s::PLI2Unbiased{T})
    out = zeros(T, size(s.x, 1), size(s.x, 2))
    for i = 1:size(s.x, 2), j = 1:size(s.x, 1)
        n = s.n[j, i]
        out[j, i] = (n * abs2(s.x[j, i]/n) - 1)/(n - 1)
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
function finish{T}(s::WPLI{T})
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
datasize(s::WPLI2Debiased, nout) = (3, nout, size(s.pairs, 2))
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
# Jammalamadaka circular correlation
#
# See Jammalamadaka, S. R., & Sengupta, A. (2001). Topics in Circular
# Statistics. World Scientific, p. 176
#
# The algorithm below is a single pass version of the algorithm in
# Jammalamadaka & Sengupta. The tests verify that it gives the same
# result.
@pairwisestat JCircularCorrelation Array{Complex{T},3}
datasize(s::JCircularCorrelation, nout) = (6, nout, size(s.pairs, 2))
@accumulatebypair JCircularCorrelation A j i x y begin
    xp = x/abs(x)
    yp = y/abs(y)
    a = xp*conj(yp)
    b = xp*yp
    A[1, j, i] += xp
    A[2, j, i] += yp
    A[3, j, i] += a - b
    A[4, j, i] += a + b
    A[5, j, i] += xp*xp
    A[6, j, i] += yp*yp
end

function finish{T}(s::JCircularCorrelation{T})
    out = zeros(T, size(s.x, 2), size(s.x, 3))
    x = s.x
    n = s.n
    for i = 1:size(out, 2), j = 1:size(out, 1)
        ni = n[j, i]
        μ = x[1, j, i]
        μ /= abs(μ)
        μ² = μ*μ
        υ = x[2, j, i]
        υ /= abs(υ)
        υ² = υ*υ
        ad = x[3, j, i]
        bd = x[4, j, i]
        expi2α = x[5, j, i]
        expi2β = x[6, j, i]
        den = (ni - real(μ²)*real(expi2α) - imag(μ²)*imag(expi2α)) *
              (ni - real(υ²)*real(expi2β) - imag(υ²)*imag(expi2β))
        # == (ni - real(μ2*expi2α))(ni - real(υ2*expi2β))
        if den > 0
            num = (real(υ)*(real(ad)*real(μ) + imag(ad)*imag(μ)) +
                   imag(υ)*(real(bd)*imag(μ) - imag(bd)*real(μ)))
            # == real(υ)*real(ad*μ) + imag(υ)*imag(bd*μ)
            out[j, i] = num/sqrt(den)
        end
    end
    out
end

#
# Circular correlation statistic of Jupp and Mardia
#
# See Jupp, P. E., & Mardia, K. V. (1980). A General Correlation
# Coefficient for Directional Data and Related Regression Problems.
# Biometrika, 67(1), 163–173. doi:10.2307/2335329
#
# The formula in terms of correlation coefficients given in the text
# is slightly wrong. The correct formula is given in:
#
# Mardia, K. V., & Jupp, P. E. (2009). Directional Statistics. John
# Wiley & Sons, p. 249. 
#
# Note that the strategy we use here to compute the correlation
# coefficients is potentially prone to catastrophic cancellation.
# However, in my simulations, this appears to be stable within the
# biologically relevant domain. If there are millions of data points or
# the circular variance is very low, you may be better off computing
# the circular correlation using canonical correlation directly.
#
@pairwisestat JMCircularCorrelation Array{T,3}
datasize(s::JMCircularCorrelation, nout) = (14, nout, size(s.pairs, 2))
@accumulatebypair JMCircularCorrelation A j i x y begin
    xp = x/abs(x)
    yp = y/abs(y)
    A[1, j, i] += real(xp)
    A[2, j, i] += imag(xp)
    A[3, j, i] += real(yp)
    A[4, j, i] += imag(yp)
    A[5, j, i] += real(xp)*real(yp)
    A[6, j, i] += real(xp)*imag(yp)
    A[7, j, i] += imag(xp)*real(yp)
    A[8, j, i] += imag(xp)*imag(yp)
    A[9, j, i] += real(xp)*imag(xp)
    A[10, j, i] += real(yp)*imag(yp)
    A[11, j, i] += abs2(real(xp))
    A[12, j, i] += abs2(imag(xp))
    A[13, j, i] += abs2(real(yp))
    A[14, j, i] += abs2(imag(yp))
end

function corp(n, xy, x, y, x2, y2)
    num = (n*xy - x*y)
    den = (n*x2 - abs2(x))*(n*y2 - abs2(y))
    (num/sqrt(den), abs2(num)/den)
end
function finish{T}(s::JMCircularCorrelation{T})
    out = zeros(T, size(s.x, 2), size(s.x, 3))
    A = s.x
    n = s.n
    @inbounds for i = 1:size(out, 2), j = 1:size(out, 1)
        ni = n[j, i]
        if ni <= 1
            out[j, i] = nan(T)
            continue
        end
        xc = A[1, j, i]
        xs = A[2, j, i]
        yc = A[3, j, i]
        ys = A[4, j, i]
        xcyc = A[5, j, i]
        xcys = A[6, j, i]
        xsyc = A[7, j, i]
        xsys = A[8, j, i]
        xcs = A[9, j, i]
        ycs = A[10, j, i]
        xc2 = A[11, j, i]
        xs2 = A[12, j, i]
        yc2 = A[13, j, i]
        ys2 = A[14, j, i]
        (ρcc, ρcc²) = corp(ni, xcyc, xc, yc, xc2, yc2)
        (ρcs, ρcs²) = corp(ni, xcys, xc, ys, xc2, ys2)
        (ρsc, ρsc²) = corp(ni, xsyc, xs, yc, xs2, yc2)
        (ρss, ρss²) = corp(ni, xsys, xs, ys, xs2, ys2)
        (ρ1, ρ1²) = corp(ni, xcs, xc, xs, xc2, xs2)
        (ρ2, ρ2²) = corp(ni, ycs, yc, ys, yc2, ys2)
        out[j, i] = (ρcc² + ρcs² + ρsc² + ρss² + 2*(ρcc*ρss + ρcs*ρsc)*ρ1*ρ2 -
                     2*(ρcc*ρcs + ρsc*ρss)*ρ2 - 2(ρcc*ρsc + ρcs*ρss)*ρ1)/((1-ρ1²)*(1-ρ2²))
    end
    out
end

#
# Entropy-based cross frequency coupling
#
# See:
# Hurtado, J. M., Rubchinsky, L. L., & Sigvardt, K. A. (2004).
# Statistical Method for Detection of Phase-Locking Episodes in Neural
# Oscillations. Journal of Neurophysiology, 91(4), 1883–1898.
# doi:10.1152/jn.00853.2003
# Tort, A. B. L., Kramer, M. A., Thorn, C., Gibson, D. J., Kubota, Y.,
# Graybiel, A. M., & Kopell, N. J. (2008). Dynamic cross-frequency
# couplings of local field potential oscillations in rat striatum and
# hippocampus during performance of a T-maze task. Proceedings of the
# National Academy of Sciences, 105(51), 20517–20522.
# doi:10.1073/pnas.0810524105
# 

type HurtadoModulationIndex{T<:Real} <: PairwiseTransformStatistic{T}
    nbins::Uint8
    pairs::Array{Int,2}
    x::Array{T,4}
    tmp_phase::Matrix{Uint8}
    tmp_amp::Matrix{T}
    n::Array{Int32,3}
    HurtadoModulationIndex() = new(18)
    HurtadoModulationIndex(nbins::Integer) =
        (nbins <= 255 || error("nbins must be <= 255"); new(nbins))
    HurtadoModulationIndex(pairs::Array{Int, 2}) = new(18, pairs)
    HurtadoModulationIndex(pairs::Array{Int, 2}, nbins::Integer) =
        (nbins <= 255 || error("nbins must be <= 255"); new(nbins, pairs))
end
HurtadoModulationIndex(args...) = HurtadoModulationIndex{Float64}(args...)

function init{T}(s::HurtadoModulationIndex{T}, nout, nchannels, ntapers, ntrials)
    if !isdefined(s, :pairs); s.pairs = allorderedpairs(nchannels); end
    s.x = zeros(T, s.nbins, nout, nout, size(s.pairs, 2))
    s.tmp_phase = zeros(Uint8, nout, nchannels)
    s.tmp_amp = zeros(T, nout, nchannels)
    s.n = zeros(Int32, s.nbins, nout, size(s.pairs, 2))
end

function accumulate(s::HurtadoModulationIndex, fftout, itaper)
    pairs = s.pairs
    tmp_phase = s.tmp_phase
    tmp_amp = s.tmp_amp
    n = s.n
    A = s.x
    nbins = s.nbins
    sz = size(fftout, 1)

    m = nbins/2pi
    for j = 1:size(fftout, 2), i = 1:size(fftout, 1)
        @inbounds a = (angle(fftout[i, j])+pi)*m
        @inbounds tmp_phase[i, j] = isnan(a) ? uint8(0) : a >= nbins ? nbins : uint8(iceil(a))
        @inbounds tmp_amp[i, j] = abs(fftout[i, j])
    end

    @inbounds begin
        for ipair = 1:size(pairs, 2)
            ch1offset = (pairs[1, ipair]-1)*sz
            ch2offset = (pairs[2, ipair]-1)*sz
            pairoffset = (ipair-1)*sz

            for ifreq1 = 1:sz
                phasebin = tmp_phase[ifreq1+ch1offset]
                phasebin != 0 || continue
                n[phasebin, ifreq1, ipair] += 1

                for ifreq2 = 1:sz
                    amp = tmp_amp[ifreq2+ch2offset]
                    !isnan(amp) || continue
                    A[phasebin, ifreq1, ifreq2, ipair] += amp
                end
            end
        end
    end
end

function finish{T}(s::HurtadoModulationIndex{T})
    A = s.x
    n = s.n
    nbins = s.nbins
    out = zeros(T, size(A, 2), size(A, 3), size(A, 4))
    tmp = zeros(typeof(zero(T)/1), nbins)
    hmax = log(float64(nbins))
    @inbounds for l = 1:size(A, 4), k = 1:size(A, 3), j = 1:size(A, 2)
        d = zero(T)/1+zero(T)/1
        for i = 1:size(A, 1)
            nv = n[i, j, l]
            d += tmp[i] = A[i, j, k, l]/nv
        end
        d = inv(d)

        h = zero(T)/1+zero(T)/1
        for i = 1:nbins
            p = tmp[i]*d
            h -= p*log(p)
        end

        out[j, k, l] = (hmax - h)/hmax
    end
    out
end

#
# Shift predictors
#
type ShiftPredictor{T<:Real,S<:PairwiseTransformStatistic} <: PairwiseTransformStatistic{T}
    stat::S
    lag::Int
    first::Array{Complex{T}, 4}
    previous::Array{Complex{T}, 4}
    buffered::Int
    pos::Int
    remaining::Int

    ShiftPredictor(s::PairwiseTransformStatistic{T}, lag::Int) = new(s, lag)
end
ShiftPredictor{T}(s::PairwiseTransformStatistic{T}, lag::Int=1) =
    ShiftPredictor{T,typeof(s)}(s, lag)

function init{T}(s::ShiftPredictor{T}, nout, nchannels, ntapers, ntrials)
    if ntrials <= s.lag
        error("Need >lag trials to generate shift predictor")
    end
    s.first = Array(Complex{T}, nout, nchannels, ntapers, s.lag)
    s.previous = Array(Complex{T}, nout, nchannels, ntapers, s.lag)
    s.buffered = 0
    s.pos = 0
    s.remaining = ntrials*ntapers
    init(s.stat, nout, nchannels, ntapers, ntrials)
end

function accumulate(s::ShiftPredictor, fftout, itaper)
    offset = size(s.previous, 1)*size(s.previous, 2)
    ntapers = size(s.previous, 3)
    bufsize = ntapers*size(s.previous, 4)

    previous = pointer_to_array(pointer(s.previous, offset*s.pos+1),
                                (size(s.previous, 1), size(s.previous, 2)), false)
    if s.remaining <= 0
        first = pointer_to_array(pointer(s.first, offset*(-s.remaining)+1),
                                 (size(s.previous, 1), size(s.previous, 2)), false)
        accumulatepairs(s.stat, first, previous, (-s.remaining % ntapers)+1)
        s.buffered -= 1
    elseif s.buffered < bufsize
        first = pointer_to_array(pointer(s.first, offset*s.buffered+1),
                                 (size(s.previous, 1), size(s.previous, 2)), false)
        copy!(first, fftout)
        copy!(previous, fftout)
        s.buffered += 1
    else
        accumulatepairs(s.stat, fftout, previous, itaper)
        copy!(previous, fftout)
    end
    s.pos = (s.pos + 1) % bufsize
    s.remaining -= 1
end

function accumulateinto!(x, n, s::ShiftPredictor, fftout, itaper)
    ret = false
    offset = size(s.previous, 1)*size(s.previous, 2)
    ntapers = size(s.previous, 3)
    bufsize = ntapers*size(s.previous, 4)

    previous = pointer_to_array(pointer(s.previous, offset*s.pos+1),
                                (size(s.previous, 1), size(s.previous, 2)), false)
    if s.remaining <= 0
        first = pointer_to_array(pointer(s.first, offset*(-s.remaining)+1),
                                 (size(s.previous, 1), size(s.previous, 2)), false)
        accumulateinternal(x, n, s.stat, first, previous, (-s.remaining % ntapers)+1)
        s.buffered -= 1
        ret = true
    elseif s.buffered < bufsize
        first = pointer_to_array(pointer(s.first, offset*s.buffered+1),
                                 (size(s.previous, 1), size(s.previous, 2)), false)
        copy!(first, fftout)
        copy!(previous, fftout)
        s.buffered += 1
    else
        accumulateinternal(x, n, s.stat, fftout, previous, itaper)
        copy!(previous, fftout)
        ret = true
    end
    s.pos = (s.pos + 1) % bufsize
    s.remaining -= 1

    ret
end

function finish(s::ShiftPredictor)
    s.remaining = 0
    while s.buffered != 0
        accumulate(s, nothing, nothing)
    end
    finish(s.stat)
end

#
# Jackknife
#

# Compute bias and variance based on jackknife surrogates
#
# truestat is the value of the statistic computed over the entire data
#     set, frequencies x channels
# surrogates are the jackknife surrogates,
#     frequencies x channels x ntrials
# n is the number of valid tapers for each data point,
#     frequencies x channels x ntrials
#
# Returns (truestat, variance, bias)
function jackknife_bias_var{T,S<:Integer}(truestat::Array{T,3}, surrogates::Array{T,4}, n::Array{S,4})
    b, v = jackknife_bias_var(reshape(truestat, stride(truestat, 3), size(truestat, 3)),
                              reshape(surrogates, stride(surrogates, 3), size(surrogates, 3), size(surrogates, 4)),
                              reshape(n, stride(n, 3), size(n, 3), size(n, 4)))
    (reshape(b, size(truestat)), reshape(v, size(truestat)))
end
function jackknife_bias_var{T,S<:Integer}(truestat::Matrix{T}, surrogates::Array{T,3}, n::Array{S,3})
    # Compute sum
    m = zeros(T, size(truestat))
    for i = 1:size(surrogates, 3), j = 1:size(surrogates, 2), k = 1:size(surrogates, 1)
        # Ignore if no tapers
        if n[k, j, i] != 0
            m[k, j] += surrogates[k, j, i]
        end
    end

    # Divide sum by number of non-zero pairs
    nz = reshape(sum(n .!= 0, 3), size(n, 1), size(n, 2))
    bias = similar(m)
    for i = 1:length(m)
        m[i] = m[i]/nz[i]
        bias[i] = (nz[i] - 1)*(m[i] - truestat[i])
    end

    # Compute variance
    variance = zeros(eltype(m), size(m))
    @inbounds begin
        for i = 1:size(surrogates, 3), j = 1:size(surrogates, 2), k = 1:size(surrogates, 1)
            if n[k, j, i] != 0
                variance[k, j] += abs2(surrogates[k, j, i] - m[k, j])
            end
        end
    end
    for i = 1:length(variance)
        variance[i] = variance[i] * (nz[i] - 1)/nz[i]
    end

    (bias, variance)
end

type JackknifeStat{T<:Real,S<:Union(PairwiseTransformStatistic,PowerSpectrum),U<:Number,N} <: TransformStatistic{T}
    stat::S

    ntapers::Int
    count::Int
    xoffset::Int
    x::Array{U,N}
    noffset::Int
    n::Array{Int32,3}
    ntapers::Int

    JackknifeStat(stat::S) = new(stat)
end
function Jackknife{T<:Real}(s::Union(PairwiseTransformStatistic{T},PowerSpectrum{T}))
    dtype = fieldtype(s, :x)
    JackknifeStat{T,typeof(s),eltype(dtype),ndims(dtype)+1}(s)
end
function Jackknife{T<:Real}(s::ShiftPredictor{T})
    dtype = fieldtype(s.stat, :x)
    JackknifeStat{T,typeof(s),eltype(dtype),ndims(dtype)+1}(s)
end

# Get the underlying statistic
motherstat{T}(s::JackknifeStat{T}) = s.stat
motherstat{T,S<:ShiftPredictor}(s::JackknifeStat{T,S}) = s.stat.stat

function init{T,S,U}(s::JackknifeStat{T,S,U}, nout, nchannels, ntapers, ntrials)
    init(s.stat, nout, nchannels, ntapers, ntrials)
    s.ntapers = ntapers
    s.count = 0

    stat = motherstat(s)
    xsize = size(stat.x)
    nsize = size(stat.n)

    s.xoffset = prod(xsize)
    s.x = zeros(U, xsize..., ntrials)

    s.noffset = prod(nsize)
    s.n = zeros(Int32, nsize..., ntrials)
end
function accumulate(s::JackknifeStat, fftout, itaper)
    # Accumulate into trial slice
    i = s.count
    x = pointer_to_array(pointer(s.x, s.xoffset*i+1), size(motherstat(s).x))
    n = pointer_to_array(pointer(s.n, s.noffset*i+1), (size(s.n, 1), size(s.n, 2)))
    accumulated = accumulateinto!(x, n, s.stat, fftout, itaper)
    s.count += (accumulated && itaper == s.ntapers)
    accumulated
end
function finish{T,S}(s::JackknifeStat{T,S})
    x = s.x
    n = s.n

    # The shift predictor requires that we continue to accumulate after
    # the final FFT has been finished
    while s.count < size(n, 3)
        for i = 1:s.ntapers
            accumulate(s, nothing, i)
        end
    end

    stat = motherstat(s)
    xsize = size(stat.x)
    nsize = size(stat.n)

    xsum = sum(x, ndims(x))

    # Subtract each x and n from the sum
    broadcast!(-, x, xsum, x)
    nsum = sum(n, 3)
    nsub = nsum .- n
    nz = reshape(sum(n .!= 0, 3), size(n, 1), size(n, 2))

    # Compute true statistic
    stat.x = squeeze(xsum, ndims(xsum))
    stat.n = squeeze(nsum, 3)
    truestat = finish(stat)

    # Compute for first surrogate
    stat.x = pointer_to_array(pointer(x, 1), xsize, false)
    stat.n = pointer_to_array(pointer(nsub, 1), nsize, false)
    surrogates = similar(truestat, size(truestat, 1), size(truestat, 2), s.count)

    # Compute statistic and mean for subsequent surrogates
    for i = 1:s.count
        stat.x = pointer_to_array(pointer(x, s.xoffset*(i-1)+1), xsize, false)
        stat.n = pointer_to_array(pointer(nsub, s.noffset*(i-1)+1), nsize, false)
        out = finish(stat)
        surrogates[:, :, i] = out
    end

    (truestat, surrogates, n)
end

# Jackknife for Coherency/Coherence
for csym in (:Coherency, :Coherence)
    sym = symbol("Jackknife$(string(csym))")
    @eval begin
        type $sym{T<:Real,S} <: TransformStatistic{T}
            psd::JackknifeStat{T,PowerSpectrum{T},T,3}
            xspec::JackknifeStat{T,S,Complex{T},3}
        end
        Jackknife{T<:Real}(s::$(csym){T}) =
            $sym{T,CrossSpectrum{T}}(Jackknife(PowerSpectrum{T}()), isdefined(s, :pairs) ? Jackknife(CrossSpectrum{T}(s.pairs)) :
                                                                                           Jackknife(CrossSpectrum{T}()))
        Jackknife{T<:Real}(s::ShiftPredictor{T,$(csym){T}}) =
            $sym{T,ShiftPredictor{T,CrossSpectrum{T}}}(Jackknife(PowerSpectrum{T}()),
                                                       isdefined(s.stat, :pairs) ? Jackknife(ShiftPredictor(CrossSpectrum{T}(s.stat.pairs), s.lag)) :
                                                                                   Jackknife(ShiftPredictor(CrossSpectrum{T}(), s.lag)))
    end
end
function init{T<:Real}(s::Union(JackknifeCoherency{T}, JackknifeCoherence{T}), nout, nchannels, ntapers, ntrials)
    init(s.psd, nout, nchannels, ntapers, ntrials)
    init(s.xspec, nout, nchannels, ntapers, ntrials)
end
function accumulate(s::Union(JackknifeCoherency, JackknifeCoherence), fftout, itaper)
    accumulate(s.psd, fftout, itaper)
    accumulate(s.xspec, fftout, itaper)
end
function finish_coherency_sp!(pairs, psd, xspec, lag)
    Base.sqrt!(psd)
    noncirc = size(xspec, 3)-lag
    for k = 1:noncirc, j = 1:size(pairs, 2)
        ch1 = pairs[1, j]
        ch2 = pairs[2, j]
        for i = 1:size(xspec, 1)
            xspec[i, j, k] = xspec[i, j, k]/(psd[i, ch1, k+lag]*psd[i, ch2, k])
        end
    end
    for k = noncirc+1:size(xspec, 3), j = 1:size(pairs, 2)
        ch1 = pairs[1, j]
        ch2 = pairs[2, j]
        for i = 1:size(xspec, 1)
            xspec[i, j, k] = xspec[i, j, k]/(psd[i, ch1, k-noncirc]*psd[i, ch2, k])
        end
    end
    xspec
end
function _finish{T<:Real,S<:ShiftPredictor}(s::Union(JackknifeCoherency{T,S}, JackknifeCoherence{T,S}))
    (truepsd, surrogatepsd, npsd) = finish(s.psd)
    (truexspec, surrogatexspec, nxspec) = finish(s.xspec)
    truecoherency = finish_coherency!(s.xspec.stat.stat.pairs, truepsd, truexspec)
    surrogatecoherency = finish_coherency_sp!(s.xspec.stat.stat.pairs, surrogatepsd, surrogatexspec, s.xspec.stat.lag)
    (truecoherency, surrogatecoherency, nxspec)
end
function _finish(s::Union(JackknifeCoherency, JackknifeCoherence))
    (truepsd, surrogatepsd, npsd) = finish(s.psd)
    (truexspec, surrogatexspec, nxspec) = finish(s.xspec)
    truecoherency = finish_coherency!(s.xspec.stat.pairs, truepsd, truexspec)
    surrogatecoherency = finish_coherency!(s.xspec.stat.pairs, surrogatepsd, surrogatexspec)
    (truecoherency, surrogatecoherency, nxspec)
end
finish(s::JackknifeCoherency) = _finish(s)
function finish(s::JackknifeCoherence)
    (truecoherency, surrogatecoherency, nxspec) = _finish(s)
    (abs(truecoherency), abs(surrogatecoherency), nxspec)
end

#
# Apply transform statistic to transformed data
#
# Data is
# channels x trials or
# frequencies x channels x trials or
# frequencies x channels x ntapers x trials or
# frequencies x time x channels x ntapers x trials
function applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},4}; trials=1:size(data,4))
    init(s, size(data, 1), size(data, 2), size(data, 3), size(data, 4))
    offset = size(data, 1)*size(data, 2)
    for j = trials, itaper = 1:size(data, 3)
            accumulate(s, pointer_to_array(pointer(data,
                                                   offset*((itaper-1)+size(data, 3)*(j-1))+1),
                                           (size(data, 1), size(data, 2))), itaper)
    end
    finish(s)
end
applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},2}; trials=1:size(data,2)) =
    vec(applystat(s, reshape(data, 1, size(data, 1), 1, size(data, 2)), trials=trials))
applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},3}; trials=1:size(data,3)) =
    applystat(s, reshape(data, size(data, 1), size(data, 2), 1, size(data, 3)), trials=trials)

drop1(x1, xs...) = xs
reshape_output(d1::Int, d2::Int, out) = reshape(out, d1, d2, drop1(size(out)...)...)
reshape_output_tuple(d1::Int, d2::Int) = ()
reshape_output_tuple(d1::Int, d2::Int, out1, out...) = tuple(reshape_output(d1, d2, out1), reshape_output_tuple(d1, d2, out...)...)
reshape_output(d1::Int, d2::Int, out::Tuple) = reshape_output_tuple(d1, d2, out...)
function applystat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},5}; trials=1:size(data,5))
    out = applystat(s, reshape(data, size(data, 1)*size(data, 2), size(data, 3),
                               size(data, 4), size(data, 5)), trials=trials)
    reshape_output(size(data, 1), size(data, 2), out)
end

#
# Apply transform statistic to permutations of transformed data
#
# Data format is as above
function permstat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},4}, nperms::Int; trials=1:size(data,4))
    p1 = doperm(s, data, trials)
    perms = similar(p1, tuple(size(p1, 1), size(p1, 2), nperms))
    perms[:, :, 1] = p1
    for i = 2:nperms
        perms[:, :, i] = doperm(s, data, trials)
    end
    perms
end
permstat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},3}, nperms::Int; trials=1:size(data,3)) =
    permstat(s, reshape(data, size(data, 1), size(data, 2), 1, size(data, 3)), nperms; trials=trials)
function permstat{T<:Real}(s::TransformStatistic{T}, data::Array{Complex{T},5}, nperms::Int; trials=1:size(data,5))
    out = permstat(s, reshape(data, size(data, 1)*size(data, 2), size(data, 3),
                               size(data, 4), size(data, 5)), nperms; trials=trials);
    reshape(out, size(data, 1), size(data, 2), size(out, 2), nperms)
end

function doperm(s, data, trials)
    init(s, size(data, 1), size(data, 2), size(data, 3), size(data, 4))

    shuffled_trials = isa(trials, Vector{Int}) ? copy(trials) : convert(Vector{Int}, trials)
    shuffle!(shuffled_trials)

    for itrial = 1:length(trials), itaper = 1:size(data, 3)
        real = pointer_to_array(pointer(data, ((trials[itrial]-1)*size(data, 3)+(itaper-1))*size(data, 2)*size(data, 1)+1),
                                (size(data, 1), size(data, 2)), false)
        shuffled = pointer_to_array(pointer(data, ((shuffled_trials[itrial]-1)*size(data, 3)+(itaper-1))*size(data, 2)*size(data, 1)+1),
                                    (size(data, 1), size(data, 2)), false)
        accumulatepairs(s, real, shuffled, itaper)
    end
    finish(s)
end
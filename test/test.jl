using FrequencyDomainAnalysis, Base.Test

const testdir = joinpath(Pkg.dir("FrequencyDomainAnalysis"), "test")

# Test dpss against dpss computed with MATLAB
d1 = dpss(128, 4)
d2 = readdlm(joinpath(testdir, "dpss128,4.txt"), '\t')
@test_approx_eq d1 d2

# Test psd output against output of pmtm for sinusoid with white noise
# This can be computed in MATLAB as pmtm(x,4,nextpow2(length(x)),1000,'unity')
input = cell(2)
output = cell(2)
for i = 1:2
	input[i] = readdlm(joinpath(testdir, "psd_test$(i)_in.txt"), '\t')[:]
	output[i] = psd(input[i], 1000, nfft=512)[:]
	truth = readdlm(joinpath(testdir, "psd_test$(i)_out.txt"), '\t')
	@test_approx_eq output[i] truth
end

# Ensure that psd gvies the same result when applied across multiple dimensions
in1 = hcat(input...)
out1 = hcat(output...)
in2 = hcat(flipud(input)...)
out2 = hcat(flipud(output)...)
@test_approx_eq psd(in1, 1000, nfft=512) out1
@test_approx_eq psd(in2, 1000, nfft=512) out2

# Test cross-spectrum
sXY1 = xspec(input[1], input[2], 1000, nfft=512)
sXY2 = xspec(input[2], input[1], 1000, nfft=512)
@test_approx_eq sXY1 conj(sXY2)
c = coherence(input[1], input[2], 1000, nfft=512)
truth = readdlm(joinpath(testdir, "coherence_mag.txt"), '\t')
@test_approx_eq c truth

# Test multiple channel functionality
(xs, s, c2) = multitaper(in1, (CrossSpectrum(), PowerSpectrum(), Coherency()), 1000, nfft=512)
@test_approx_eq s out1
@test_approx_eq xs sXY1
truth = readdlm(joinpath(testdir, "coherence_phi.txt"), '\t')[2:end-1]
@test_approx_eq abs(c2) c
@test_approx_eq angle(c2[2:end-1]) truth

# Test mtfft and applystat
ft = mtfft(in1, 1000, nfft=512)
@test_approx_eq applystat(PowerSpectrum(), ft) out1
@test_approx_eq applystat(CrossSpectrum(), ft) sXY1
@test_approx_eq applystat(Coherence(), ft) c

# Test power spectrum variance
varin = reshape(input[1], 43, 1, 7)
ft = mtfft(varin, 1000, nfft=512)
@test_approx_eq applystat(PowerSpectrumVariance(), ft) var(mean(abs2(ft), 3), 4)

# Test PLV and PPC
angles = [
	2.439564585801219,
	3.0190627944596296,
	1.9558533611578697,
	2.9276497655265747,
	-2.668278147116921,
	2.9670038532462017,
	-2.489899551868197,
	-2.7098277543134612,
	-2.0699115143373175,
	-1.6574845096744018,
	-1.7483177390187856,
	3.0954095279265252,
	2.8604385096072225,
	-3.04237881355147,
	2.6904076043411562
]
true_plv = abs(mean(exp(im*angles)))
true_ppc = 1-var(exp(im*angles)) # yep.

period = 32
nperiods = 16
band = nperiods + 1
x = 0:2*pi/period:2*pi*nperiods-2*pi/period
signal1 = repeat(cos(x), inner=[1, 1, length(angles)])
signal2 = cat(3, [cos(x + a) for a in angles]...)
plv_signals = [signal1 signal2]

coh, plv, ppc = multitaper(plv_signals, (Coherence(), PLV(), PPC()), tapers=ones(period*nperiods))

@test_approx_eq coh[band] true_plv
@test_approx_eq plv[band] true_plv
@test_approx_eq ppc[band] true_ppc

# Test NaN handling
ft1 = mtfft(plv_signals, tapers=ones(period*nperiods))
ft2 = copy(ft1)
ft2[5, 2, 1] = NaN

expected = applystat(PowerSpectrum(), ft1)
expected[5, 2] = applystat(PowerSpectrum(), ft1[:, :, 2:end])[5, 2]
out = applystat(PowerSpectrum(), ft2)
@test_approx_eq out expected

out = applystat(CrossSpectrum(), ft2)
expected = applystat(CrossSpectrum(), ft1)
expected[5, 1] = applystat(CrossSpectrum(), ft1[:, :, 2:end])[5, 1]
@test_approx_eq out expected

out = applystat(Coherence(), ft2)
expected = applystat(Coherence(), ft1)
expected[5, 1] = applystat(Coherence(), ft1[:, :, 2:end])[5, 1]
@test_approx_eq out expected

ft2[5, 2, :] = NaN
@test findn(isnan(applystat(PowerSpectrum(), ft2))) == ([5],[2])
@test findn(isnan(real(applystat(CrossSpectrum(), ft2)))) == ([5],[1])
@test findn(isnan(applystat(Coherence(), ft2))) == ([5],[1])

# Test jackknife
jn = multitaper(plv_signals, Jackknife(PLV()), tapers=ones(period*nperiods))
@test_approx_eq jn[1] plv
estimates = zeros(size(plv, 1), size(plv, 2), size(plv_signals, 3))
for i = 1:size(plv_signals, 3)
	estimates[:, :, i] = multitaper(plv_signals[:, :, [1:i-1, i+1:size(plv_signals,3)]],
		                            PLV(), tapers=ones(period*nperiods))
end
@test_approx_eq jn[2] var(estimates, 3)
@test_approx_eq jn[3] (size(plv_signals, 3)-1)*(mean(estimates, 3) - plv)

# Test shift predictor
x = 0:63
signal = zeros(length(x), 1, 35)
for i = 1:size(signal, 3)
	signal[:, 1, i] = cos(0.2pi*x+rand()*2pi)
end
c = multitaper([signal signal], Coherence())
@test_approx_eq c 1
for lag = 1:5
	t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], Coherence())
	sp = multitaper([signal signal], ShiftPredictor(Coherence(), lag))
	@test_approx_eq t sp
end

# Test jackknifed shift predictor
for lag = 1:5
	t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], Jackknife(PLV()))
	sp = multitaper([signal signal], Jackknife(ShiftPredictor(PLV(), lag)))
	@test_approx_eq t[1] sp[1]
	@test_approx_eq t[2] sp[2]
	@test_approx_eq t[3] sp[3]
end

# Test permutations
#
# There's no good test for whether the output is "correct," but we test
# a case where the permuted output should be unity and a case where it
# shouldn't
x = 0:63
ft = mtfft(repeat(signal[:, 1, 1], inner=[1, 2, 35]))
perms = permstat(Coherence(), ft, 10)
@test_approx_eq perms 1
ft = mtfft([signal signal])
perms = permstat(Coherence(), ft, 10)
@test all(perms .!= 1)

# Test Morlet wavelet bases
#
# Reference implementation is from Torrence, C., and G. P. Compo. “A
# Practical Guide to Wavelet Analysis.” Bulletin of the American
# Meteorological Society 79, no. 1 (1998): 61–78.
#
# Our bases differ from Torrence and Compo's by a factor corresponding
# to the length of the FFT, which eliminates the need to normalize
# the output of the inverse FFT.
d1 = [wavebases(MorletWavelet([0.1], 5), 1024) * 1024; zeros(511, 1)]
d1[1, :] = 0 # Torrence and Compo don't use the zero frequency
             # component because they subtract the signal mean in the
             # time domain.
d2 = readdlm(joinpath(testdir, "morlet_bases_f_0.1_k0_5.txt"))
@test_approx_eq d1 d2

# Test Morlet wavelet by comparing output for a 0.1 Hz oscillation
# embedded in white noise
test_in = readdlm(joinpath(testdir, "wavelet_test_in.txt"))[:]
foi = readdlm(joinpath(testdir, "wavelet_test_foi.txt"))[:]
d1 = cwt(test_in - mean(test_in), MorletWavelet(foi, 5))
d2 = complex(readdlm(joinpath(testdir, "wavelet_test_out_re.txt"), '\t'), readdlm(joinpath(testdir, "wavelet_test_out_im.txt"), '\t'))
coi_periods = readdlm(joinpath(testdir, "wavelet_test_coi.txt"))[:]
for j = 1:length(foi)
	period = 1/foi[j]
	for i = 1:length(coi_periods)
		@test isnan(real(d1[i, j])) == (1/foi[j] > coi_periods[i])
	end
end
d2[isnan(real(d1))] = NaN
@test_approx_eq d1 d2

# Test missing data handling in wavelet transform
y = ones(1000)
y[499] = 0
y[500] = 1/3
y[501] = 2/3
foi = 2.^(0:0.25:7)
w = MorletWavelet(foi, 5)
cois = iceil(wavecoi(w, 1000))
z1 = cwt(y, w, 1000)
y[500] = NaN
z2 = cwt(y, w, 1000)
for k = 1:length(foi), i = 1:length(y)
	@test isnan(real(z2[i, k])) == (i <= cois[k] || i >= length(y) - cois[k] + 1 ||
		                            500 - cois[k] <= i <= 500 + cois[k])
end
@test_approx_eq z1[!isnan(real(z2))] z2[!isnan(real(z2))]
y[501] = NaN
z2 = cwt(y, w, 1000)
for k = 1:length(foi), i = 1:length(y)
	@test isnan(real(z2[i, k])) == (i <= cois[k] || i >= length(y) - cois[k] + 1 ||
		                            500 - cois[k] <= i <= 501 + cois[k])
end
@test_approx_eq z1[!isnan(real(z2))] z2[!isnan(real(z2))]

# Test pxcorr
x = [rand() > 0.5 for i = 1:50]
y = [rand() > 0.5 for i = 1:50]
@test_approx_eq xcorr(x, y) pxcorr(find(x), find(y), -49:49)

# TODO PLI, PLI2Unbiased, WPLI, WPLI2Debiased,
#      spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1,
#      pfppc2
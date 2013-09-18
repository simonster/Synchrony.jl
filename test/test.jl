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
	output[i] = psd(input[i], fs=1000)[:]
	truth = readdlm(joinpath(testdir, "psd_test$(i)_out.txt"), '\t')
	@test_approx_eq output[i] truth
end

# Ensure that psd gvies the same result when applied across multiple dimensions
in1 = hcat(input...)
out1 = hcat(output...)
in2 = hcat(flipud(input)...)
out2 = hcat(flipud(output)...)
@test_approx_eq psd(in1, fs=1000) out1
@test_approx_eq psd(in2, fs=1000) out2

# Test cross-spectrum
sXY1 = xspec(input[1], input[2]; fs=1000)
sXY2 = xspec(input[2], input[1]; fs=1000)
@test_approx_eq sXY1 conj(sXY2)
c = coherence(input[1], input[2], fs=1000)
truth = readdlm(joinpath(testdir, "coherence_mag.txt"), '\t')
@test_approx_eq c truth

# Test multiple channel functionality
in3 = hcat(input[1], input[2])
(xs, s, c2) = multitaper(in3, (CrossSpectrum(), PowerSpectrum(), Coherency()), fs=1000)
@test_approx_eq s out1
@test_approx_eq xs sXY1
truth = readdlm(joinpath(testdir, "coherence_phi.txt"), '\t')[2:end-1]
@test_approx_eq abs(c2) c
@test_approx_eq angle(c2[2:end-1]) truth

# Test shift predictor
x = 0:63
signal = zeros(length(x), 1, 35)
for i = 1:size(signal, 3)
	signal[:, i] = cos(0.2pi*x+rand()*2pi)
end
c = multitaper([signal signal], Coherence(), nfft=64)
@test_approx_eq c 1
t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], 1)]], Coherence(), nfft=64)
sp = multitaper([signal signal], ShiftPredictor(Coherence()), nfft=64)
@test_approx_eq t sp

# Test Morlet wavelet bases
d1 = [convert(Array{Float64}, wavebases(MorletWavelet([0.1], 5), 1024)); zeros(511, 1)]
d2 = readdlm(joinpath(testdir, "morlet_bases_f_0.1_k0_5.txt"))
@test_approx_eq d1 d2

# Test Morlet wavelet by comparing output for a 0.1 Hz oscillation
# embedded in white noise
#
# Reference implementation is from Torrence, C., and G. P. Compo. “A
# Practical Guide to Wavelet Analysis.” Bulletin of the American
# Meteorological Society 79, no. 1 (1998): 61–78.
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

# TODO PLV, PPC, PLI, PLI2Unbiased, WPLI, WPLI2Debiased,
#      spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1,
#      pfppc2
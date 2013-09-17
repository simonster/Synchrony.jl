using FrequencyDomainAnalysis, Base.Test

# Test Morlet wavelet bases
d1 = convert(Array{Float64}, wavebases(MorletWavelet([0.1], 5), 1024))
d2 = readdlm("morlet_bases_f_0.1_k0_5.txt")
@test_approx_eq d1 d2

# Test dpss against dpss computed with MATLAB
d1 = dpss(128, 4)
d2 = readdlm("dpss128,4.txt", '\t')
@test_approx_eq d1 d2

# Test psd output against output of pmtm for sinusoid with white noise
# This can be computed in MATLAB as pmtm(x,4,nextpow2(length(x)),1000,'unity')
input = cell(2)
output = cell(2)
for i = 1:2
	input[i] = readdlm("test$(i)_in.txt", '\t')[:]
	output[i] = psd(input[i], fs=1000)[:]
	truth = readdlm("test$(i)_out.txt", '\t')
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
truth = readdlm("coherence_mag.txt", '\t')
@test_approx_eq c truth

# Test multiple channel functionality
in3 = hcat(input[1], input[2])
(xs, s, c2) = multitaper(in3, (CrossSpectrum(), PowerSpectrum(), Coherency()), fs=1000)
@test_approx_eq s out1
@test_approx_eq xs sXY1
truth = readdlm("coherence_phi.txt", '\t')[2:end-1]
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
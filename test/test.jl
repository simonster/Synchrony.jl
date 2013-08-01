using FrequencyDomainAnalysis

# Test dpss against dpss computed with MATLAB
d1 = dpss(128, 4)
d2 = readdlm("dpss128,4.txt", '\t')
@assert all(abs(d1 - d2) .< sqrt(eps()))

# Test psd output against output of pmtm for sinusoid with white noise
# This can be computed in MATLAB as pmtm(x,4,nextpow2(length(x)),1000,'unity')
input = cell(2)
output = cell(2)
for i = 1:2
	input[i] = readdlm("test$(i)_in.txt", '\t')[:]
	output[i] = psd(input[i], fs=1000)[:]
	truth = readdlm("test$(i)_out.txt", '\t')
	@assert max(abs(output[i] - truth)) < 7*eps()
end

# Ensure that psd gvies the same result when applied across multiple dimensions
in1 = hcat(input...)
out1 = hcat(output...)
in2 = hcat(flipud(input)...)
out2 = hcat(flipud(output)...)
@assert psd(in1, fs=1000) == out1
@assert psd(in2, fs=1000) == out2

# Test cross-spectrum
sXY1 = xspec(input[1], input[2]; fs=1000)
sXY2 = xspec(input[2], input[1]; fs=1000)
@assert sXY1 == conj(sXY2)
c = coherence(input[1], input[2], fs=1000)
truth = readdlm("coherence_mag.txt", '\t')
@assert max(abs(c - truth)) < sqrt(eps())

# Test multiple channel functionality
in3 = hcat(input[1], input[2])
(xs, s, c2) = multitaper(in3, (CrossSpectrum(), PowerSpectrum(), Coherency()), fs=1000)
@assert max(abs(s - out1)) < 7*eps()
@assert max(abs(xs - sXY1)) < 7*eps()
truth = readdlm("coherence_phi.txt", '\t')[2:end-1]
@assert max(abs(c2) - c) <= eps()
@assert max(abs(angle(c2[2:end-1]) - truth)) < sqrt(eps())
using Multitaper

# Test dpss against dpss computed with MATLAB
d1 = dpss(128, 4)
d2 = readdlm("dpss128,4.txt", '\t')
@assert all(abs(d1 - d2) .< sqrt(eps()))

# Test psd output against output of pmtm for sinusoid with white noise
# This can be computed in MATLAB as pmtm(x,4,nextpow2(length(x)),1000,'unity')
input = cell(2)
output = cell(2)
for i = 1:2
	input[i] = readdlm("test$(i)_in.txt", '\t')
	output[i] = psd(input[i], fs=1000)
	truth = readdlm("test$(i)_out.txt", '\t')
	@assert all(abs(output[i] - truth) .< 7*eps())
end

# Ensure that psd gvies the same result when applied across multiple dimensions
in1 = hcat(input...)
out1 = hcat(output...)
in2 = hcat(flipud(input)...)
out2 = hcat(flipud(output)...)
@assert psd(in1, fs=1000) == out1
@assert psd(in2, fs=1000) == out2

# Test cross-spectrum
(sXY, sXX, sYY) = xspec(in1, in2, fs=1000)
@assert sXX == out1
@assert sYY == out2
@assert sXY[:, 1] == conj(sXY[:, 2])

# Test multiple channel cross-spectrum
(xs, s) = xspec(cat(3, input[1], input[2]), fs=1000, trialavg=false)
@assert all(abs(squeeze(s, 2) - out1) .< 7*eps())
@assert all(abs(xs[:] - sXY[:, 1]) .< 7*eps())

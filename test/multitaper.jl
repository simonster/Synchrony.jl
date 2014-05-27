using Synchrony, Base.Test
datadir = joinpath(Pkg.dir("Synchrony"), "test", "data")

# Test psd output against output of pmtm for sinusoid with white noise
# This can be computed in MATLAB as pmtm(x,4,nextpow2(length(x)),1000,'unity')
input = cell(2)
output = cell(2)
for i = 1:2
    input[i] = float64(readdlm(joinpath(datadir, "psd_test$(i)_in.txt"), '\t'))[:]
    output[i] = psd(input[i], 1000, nfft=512)[:]
    truth = float64(readdlm(joinpath(datadir, "psd_test$(i)_out.txt"), '\t'))
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

# Test coherence
c = coherence(input[1], input[2], 1000, nfft=512)
truth = float64(readdlm(joinpath(datadir, "coherence_mag.txt"), '\t'))
@test_approx_eq c truth

# Test multiple channel functionality
(xs, s, c2) = multitaper(in1, (CrossSpectrum(), PowerSpectrum(), Coherency()), 1000, nfft=512)
@test_approx_eq s out1
@test_approx_eq xs sXY1
truth = float64(readdlm(joinpath(datadir, "coherence_phi.txt"), '\t')[2:end-1])
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

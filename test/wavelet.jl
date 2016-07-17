using Synchrony, Base.Test
datadir = joinpath(Pkg.dir("Synchrony"), "test", "data")

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
d2 = map(Float64, readdlm(joinpath(datadir, "morlet_bases_f_0.1_k0_5.txt")))
@test_approx_eq d1 d2

# Test Morlet wavelet by comparing output for a 0.1 Hz oscillation
# embedded in white noise
test_in = map(Float64, readdlm(joinpath(datadir, "wavelet_test_in.txt"))[:])
foi = map(Float64, readdlm(joinpath(datadir, "wavelet_test_foi.txt"))[:])
d1 = cwt(test_in, MorletWavelet(foi, 5))
d2 = complex(map(Float64, readdlm(joinpath(datadir, "wavelet_test_out_re.txt"), '\t')),
             map(Float64, readdlm(joinpath(datadir, "wavelet_test_out_im.txt"), '\t')))
coi_periods = map(Float64, readdlm(joinpath(datadir, "wavelet_test_coi.txt"))[:])
for j = 1:length(foi)
    period = 1/foi[j]
    for i = 1:length(coi_periods)
        @test isnan(real(d1[i, j])) == (period > coi_periods[i]) || abs(period - coi_periods[i]) <= eps(coi_periods[i])
    end
end
d2[isnan(real(d1))] = NaN
@test_approx_eq d1 d2

# Test tstd
for w in (MorletWavelet([0.0003]), MorseWavelet([0.0003], 8, 3))
    wb = [wavebases(w, 100000, 3); zeros(49999)]
    p = abs2(ifftshift(bfft(wb)))
    x = (0:length(p)-1)/3
    μ = sum(p.*x)
    σ = sqrt(sum(p.*abs2(x.-μ)))
    @test_approx_eq_eps σ tstd(w) σ*1e-5
end

# Test fstd
for w in (MorletWavelet([0.75]), MorseWavelet([0.75], 8, 3))
    x = frequencies(100000, 3)[1]
    p = abs2(wavebases(w, 100000, 3))*100000
    μ = sum(p.*x)
    σ = sqrt(sum(p.*abs2(x.-μ)))
    @test_approx_eq_eps σ fstd(w) σ*1e-5
end

# Test missing data handling in wavelet transform
y = ones(1000)
y[499] = 0
y[500] = 1/3
y[501] = 2/3
foi = 2.^(0:0.25:7)
w = MorletWavelet(foi, 5)
cois = ceil(Int, tstd(w)*2*1000)
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

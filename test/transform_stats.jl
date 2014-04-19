using Synchrony, NumericExtensions, Base.Test, CrossDecomposition

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
x = (0:2/period:2*nperiods-2/period)*pi
signal1 = repeat(cos(x), inner=[1, 1, length(angles)])
signal2 = cat(3, [cos(x + a) for a in angles]...)
plv_signals = [signal1 signal2]

coh, plv, ppc, ccor = multitaper(plv_signals, (Coherence(), PLV(), PPC(),
                                               JCircularCorrelation()),
                                        tapers=ones(period*nperiods))

@test_approx_eq coh[band] true_plv
@test_approx_eq plv[band] true_plv
@test_approx_eq ppc[band] true_ppc
@test_approx_eq ccor[band] 0.0

# Test Jammalamadaka circular correlation
angles = [
 -2.386402923969883 0.7120356435187083
 -1.2179485025557881 -0.5107153930236978
 0.28658754714190227 -0.5752672716762053
 -0.9331371629333169 2.1870089874804917
 -0.8995711841426721 5.179928197813174
 0.8887003244657483 0.5249512145071122
 -0.9390694729799546 2.1193361160731787
 2.4997562944878062 4.444408123844822
 -0.15260607000620233 0.32401343526930065
 -1.58821218340456 3.838922093624558
 0.3300644770555941 3.612382182527298
 -0.9159071492390857 1.3286113019566637
 -1.2527393658061554 1.0733970760561626
 -4.364282749017247 1.9318390452682834
 -2.102397073673845 -1.1008030808360818
]
# adiff = randn(1000000)
# angles = [randn(1000000).+adiff randn(1000000).+adiff.+pi/2]

expi = exp(im*angles)

abar, bbar = angle(sum(expi, 1))
aabar = sin(angles[:, 1] .- abar)
bbbar = sin(angles[:, 2] .- bbar)
true_ccor = sum(aabar.*bbbar)/sqrt(sum(abs2(aabar)).*sum(abs2(bbbar)))

@test_approx_eq applystat(JCircularCorrelation(), expi.')[1] true_ccor

# Test Jupp and Mardia circular correlation
am = [real(expi[:, 1]) imag(expi[:, 1])]
bm = [real(expi[:, 2]) imag(expi[:, 2])]
true_ccor = sum(abs2(cor(canoncor(am, bm))))

# println((cor(real(expi[:, 1]), real(expi[:, 2])),
#          cor(real(expi[:, 1]), imag(expi[:, 2])),
#          cor(imag(expi[:, 1]), real(expi[:, 2])),
#          cor(imag(expi[:, 1]), imag(expi[:, 2])),
#          cor(real(expi[:, 1]), imag(expi[:, 1])),
#          cor(real(expi[:, 2]), imag(expi[:, 2]))))
@test_approx_eq applystat(JMCircularCorrelation(), expi.')[1] true_ccor

# Test NaN handling
ft1 = mtfft(plv_signals, tapers=ones(period*nperiods))
ft2 = copy(ft1)
ft2[5, 2, 1] = NaN

expected = applystat(PowerSpectrum(), ft1)
expected[5, 2] = applystat(PowerSpectrum(), ft1[:, :, 2:end])[5, 2]
out = applystat(PowerSpectrum(), ft2)
@test_approx_eq out expected

for stat in (CrossSpectrum, Coherence, PLV)
    out = applystat(stat(), ft2)
    expected = applystat(stat(), ft1)
    expected[5, 1] = applystat(stat(), ft1[:, :, 2:end])[5, 1]
    @test_approx_eq out expected
end

ft2[5, 2, :] = NaN
@test findn(isnan(applystat(PowerSpectrum(), ft2))) == ([5],[2])
for stat in (CrossSpectrum, Coherence, PLV)
    @test findn(isnan(real(applystat(stat(), ft2)))) == ([5],[1])
end

# Test jackknife
for (stat, trueval) in ((Coherence, coh), (PLV, plv))
    jn = multitaper(plv_signals, Jackknife(stat()), tapers=ones(period*nperiods))
    (bias, variance) = jackknife_bias_var(jn...)
    @test_approx_eq jn[1] trueval
    estimates = zeros(size(plv, 1), size(plv, 2), size(plv_signals, 3))
    for i = 1:size(plv_signals, 3)
        estimates[:, :, i] = multitaper(plv_signals[:, :, [1:i-1, i+1:size(plv_signals,3)]],
                                        stat(), tapers=ones(period*nperiods))
    end
    @test_approx_eq bias (size(plv_signals, 3)-1)*(mean(estimates, 3) - trueval)
    @test_approx_eq variance sum(abs2(estimates .- mean(estimates, 3)), 3)*(size(plv_signals, 3)-1)/size(plv_signals, 3)
end

# Test shift predictor
x = 0:63
signal = zeros(length(x), 1, 35)
for i = 1:size(signal, 3)
    signal[:, 1, i] = cospi(0.2*x+rand()*2)
end
c = multitaper([signal signal], Coherence())
@test_approx_eq c ones(size(c))
for lag = 1:5
    t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], Coherence())
    sp = multitaper([signal signal], ShiftPredictor(Coherence(), lag))
    @test_approx_eq t sp
end

# Test jackknifed shift predictor
for stat in (Coherence, PLV), lag = 1:5
    t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], Jackknife(stat()))
    sp = multitaper([signal signal], Jackknife(ShiftPredictor(stat(), lag)))
    @test_approx_eq t[1] sp[1]
    (tbias, tvar) = jackknife_bias_var(t...)
    (spbias, spvar) = jackknife_bias_var(sp...)
    @test_approx_eq tbias spbias
    @test_approx_eq tvar spvar
end

# Test permutations
#
# There's no good test for whether the output is "correct," but we test
# a case where the permuted output should be unity and a case where it
# shouldn't
x = 0:63
ft = mtfft(repeat(signal[:, 1, 1], inner=[1, 2, 35]))
perms = permstat(Coherence(), ft, 10)
@test_approx_eq perms ones(size(perms))
ft = mtfft([signal signal])
perms = permstat(Coherence(), ft, 10)
@test all(perms .!= 1)

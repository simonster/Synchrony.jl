using Synchrony2, Base.Test, CrossDecomposition

# Tests for statistics determined by cross spectrum
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
r = [
    0.5222318110603368,
    0.29671462136718185,
    0.0766100484932224,
    1.3353190416626122,
    1.2002648555961346,
    1.0636420608553432,
    1.745582800792132,
    0.7810048670915002,
    1.5825305843214879,
    1.3754091313969945,
    1.997253024492295,
    1.239976187334439,
    1.4558221123423296,
    1.6855505422703576,
    0.6713539625790665
]

expangles = exp(im*angles)
expcoef = r.*expangles
true_coh = mean(expcoef)./sqrt(mean(abs2(r)))
true_meanphasediff = mean(expangles)
true_ppc = 1-var(expangles)
true_pli = mean(sign(angles))
true_pli2unbiased = (length(angles)*abs2(mean(sign(angles)))-1)/(length(angles)-1)
true_wpli = abs(mean(imag(expcoef)))./mean(abs(imag(expcoef)))
pairs = Float64[]
for i1 = 1:length(angles), i2 = 1:length(angles)
    i1 != i2 || continue
    push!(pairs, imag(expcoef[i1])*imag(expcoef[i2]))
end
true_wpli2debiased = mean(pairs)./mean(abs(pairs))

oneinput = ones(Complex128, length(angles))
csinput = [oneinput expcoef]

# Single input
for (stat, output) in ((Coherency(), true_coh),
                       (Coherence(), abs(true_coh)),
                       (MeanPhaseDiff(), true_meanphasediff),
                       (PLV(), abs(true_meanphasediff)),
                       (PPC(), true_ppc),
                       (PLI(), true_pli),
                       (PLI2Unbiased(), true_pli2unbiased),
                       (WPLI(), true_wpli),
                       (WPLI2Debiased(), true_wpli2debiased))
    @test_approx_eq computestat(stat, csinput)[1, 2] output
    @test_approx_eq computestat(stat, oneinput, expcoef)[1] output
    @test_approx_eq Base.LinAlg.copytri!(computestat(stat, csinput), 'U', true) computestat(stat, csinput, csinput)
end

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

@test_approx_eq computestat(JammalamadakaR(), expi)[1, 2] true_ccor
@test_approx_eq computestat(JammalamadakaR(), expi[:, 1], expi[:, 2])[1] true_ccor

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
@test_approx_eq computestat(JuppMardiaR(), expi)[1, 2] true_ccor
@test_approx_eq computestat(JuppMardiaR(), expi[:, 1], expi[:, 2])[1] true_ccor

# Test Hurtado modulation index
# Create two signals with some phase-amplitude coupling
function dumbcfc(x, y, nbins)
    angles = angle(x)
    amps = abs(y)
    bins = linspace(-pi, pi, nbins+1)
    mean_amp = [mean(amps[bins[i] .<= angles .<= bins[i+1]]) for i = 1:nbins]
    pj = mean_amp/sum(mean_amp)
    h = -sum(pj.*log(pj))
    hmax = log(nbins)
    (hmax - h)/hmax
end
s1 = exp(im*2pi*[0.05:.1:10;]).*repmat([0.05:.1:1;], 10)
s1s = shuffle(s1)
s2 = exp(im*2pi*[1/18:1/9:11+1/18;]).*repmat([0.99; ones(9)*0.01], 10)

out = computestat(HurtadoModulationIndex(10), [s1 s2 s1s], [s1 s2])
@test_approx_eq out [dumbcfc(s1, s1, 10) dumbcfc(s1, s2, 10)
                     dumbcfc(s2, s1, 10) dumbcfc(s2, s2, 10)
                     dumbcfc(s1s, s1, 10) dumbcfc(s1s, s2, 10)]

# Test UniformScores
d = [50, 120, 192, 210, 220, 250, 262, 291, 292, 320, 321, 340,
     0, 20, 40, 60, 160, 171, 200, 221, 270, 294, 341, 350,
     10, 11, 21, 22, 31, 32, 41, 150, 151, 152, 170, 190, 293,
     30, 70, 110, 172, 180, 191, 240, 251, 260, 261, 290, 351]
groups = [fill(1, 12); fill(2, 12); fill(3, 13); fill(4, 12)]
v = computestat(UniformScores(groups), Complex128[rand(length(d)).*cis(d/180*pi) zeros(length(d))])
@test_approx_eq_eps v[1, 2] 12.81 0.01

csinput = complex(randn(25, 3), randn(25, 3))
nmulti = 5

# Generate bootstrap weights
bsindices = rand(1:size(csinput, 1), size(csinput, 1), 10)
permindices = Permutation(Coherence(), size(csinput, 1), 10).indices
groups = Vector{Tuple{Int,Int}}[[(1, 2)], [(3, 2)], [(1, 2), (2, 3)]]

# Single input tests of nested stats
for stat in [Coherence(), Coherency(), MeanPhaseDiff(), PLV(), PPC(), PLI(), PLI2Unbiased(), WPLI(), WPLI2Debiased(), JammalamadakaR(), JuppMardiaR()]
    # Test GroupMean
    trueval = computestat(stat, csinput)
    gmstat = GroupMean(stat, size(csinput, 2), groups)
    gm = computestat(gmstat, csinput)
    @test_approx_eq gm [trueval[1, 2], trueval[2, 3], (trueval[1, 2] + trueval[2, 3])/2]

    # Test JackknifeSurrogates
    estimates = zeros(eltype(trueval), size(csinput, 1), size(csinput, 2), size(csinput, 2))
    for i = 1:size(csinput, 1)
        estimates[i, :, :] = computestat(stat, csinput[[1:i-1; i+1:size(csinput, 1)], :])
    end
    jnvar = squeeze(var(estimates, 1, corrected=false), 1)*(size(csinput, 1)-1)
    jn = computestat(JackknifeSurrogates(stat), csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq_eps jackknife_bias(jn) (size(csinput, 1)-1)*(squeeze(mean(estimates, 1), 1) - trueval) sqrt(eps())
    @test_approx_eq jackknife_var(jn) jnvar

    # Test Jackknife
    a = Jackknife(stat)
    jn = computestat(Jackknife(stat), csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq jn.var jnvar

    # Test Jackknife{GroupMean}
    jn = computestat(Jackknife(gmstat), csinput)
    @test_approx_eq jn.trueval [trueval[1, 2], trueval[2, 3], (trueval[1, 2] + trueval[2, 3])/2]
    est2 = mean([estimates[:, 1, 2] estimates[:, 2, 3]], 2)
    @test_approx_eq jn.var [jnvar[1, 2], jnvar[2, 3], var(est2, corrected=false)*(size(csinput, 1)-1)]

    # Test MultiJackknifeSurrogates
    estimates = zeros(eltype(trueval), div(size(csinput, 1), nmulti), size(csinput, 2), size(csinput, 2))
    for i = 1:size(estimates, 1)
        estimates[i, :, :] = computestat(stat, csinput[[1:(i-1)*nmulti; i*nmulti+1:size(csinput, 1)], :])
    end
    jnvar = squeeze(var(estimates, 1, corrected=false), 1)*(size(estimates, 1)-1)
    jn = computestat(MultiJackknifeSurrogates(stat, nmulti), csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq_eps jackknife_bias(jn) (size(estimates, 1)-1)*(squeeze(mean(estimates, 1), 1) - trueval) sqrt(eps())
    @test_approx_eq jackknife_var(jn) jnvar

    # Test Bootstrap
    estimates = zeros(eltype(trueval), size(csinput, 2), size(csinput, 2), size(bsindices, 2))
    for i = 1:size(bsindices, 2)
        estimates[:, :, i] = computestat(stat, csinput[bsindices[:, i], :])
    end
    bs = computestat(Bootstrap(stat, bsindices), csinput)
    @test_approx_eq bs estimates

    # Test Bootstrap{GroupMean}
    bs = computestat(Bootstrap(gmstat, bsindices), csinput)
    @test_approx_eq bs squeeze([estimates[1, 2, :]; estimates[2, 3, :]; (estimates[1, 2, :] + estimates[2, 3, :])/2], 2)

    # Test Permutation
    estimates = zeros(eltype(trueval), size(csinput, 2), size(csinput, 2), size(permindices, 2))
    for i = 1:size(permindices, 2)
        estimates[:, :, i] = computestat(stat, csinput[permindices[:, i], :], csinput)
    end
    perms = computestat(Permutation(stat, permindices), csinput)
    @test_approx_eq perms estimates

    # Test Permutation{GroupMean}
    bs = computestat(Permutation(gmstat, permindices), csinput)
    @test_approx_eq bs squeeze([estimates[1, 2, :]; estimates[2, 3, :]; (estimates[1, 2, :] + estimates[2, 3, :])/2], 2)
end

# Two input tests of nested stats
for stat in [Coherence(), Coherency(), MeanPhaseDiff(), PLV(), PPC(), PLI(), PLI2Unbiased(), WPLI(), WPLI2Debiased(), JammalamadakaR(), JuppMardiaR(), HurtadoModulationIndex(5)]
    # Test GroupMean
    trueval = computestat(stat, csinput, csinput)
    gmstat = GroupMean(stat, size(csinput, 2), groups)
    gm = computestat(gmstat, csinput, csinput)
    @test_approx_eq gm [trueval[1, 2], trueval[2, 3], (trueval[1, 2] + trueval[2, 3])/2]

    # Test JackknifeSurrogates
    estimates = zeros(eltype(trueval), size(csinput, 1), size(csinput, 2), size(csinput, 2))
    for i = 1:size(csinput, 1)
        estimates[i, :, :] = computestat(stat, csinput[[1:i-1; i+1:size(csinput, 1)], :], csinput[[1:i-1; i+1:size(csinput, 1)], :])
    end
    jnvar = squeeze(var(estimates, 1, corrected=false), 1)*(size(csinput, 1)-1)
    jn = computestat(JackknifeSurrogates(stat), csinput, csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq_eps jackknife_bias(jn) (size(csinput, 1)-1)*(squeeze(mean(estimates, 1), 1) - trueval) sqrt(eps())
    @test_approx_eq jackknife_var(jn) jnvar

    # Test Jackknife
    a = Jackknife(stat)
    jn = computestat(Jackknife(stat), csinput, csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq jn.var jnvar

    # Test Jackknife{GroupMean}
    jn = computestat(Jackknife(gmstat), csinput, csinput)
    @test_approx_eq jn.trueval [trueval[1, 2], trueval[2, 3], (trueval[1, 2] + trueval[2, 3])/2]
    est2 = mean([estimates[:, 1, 2] estimates[:, 2, 3]], 2)
    @test_approx_eq jn.var [jnvar[1, 2], jnvar[2, 3], var(est2, corrected=false)*(size(csinput, 1)-1)]

    # Test MultiJackknifeSurrogates
    estimates = zeros(eltype(trueval), div(size(csinput, 1), nmulti), size(csinput, 2), size(csinput, 2))
    for i = 1:size(estimates, 1)
        estimates[i, :, :] = computestat(stat, csinput[[1:(i-1)*nmulti; i*nmulti+1:size(csinput, 1)], :], csinput[[1:(i-1)*nmulti; i*nmulti+1:size(csinput, 1)], :])
    end
    jnvar = squeeze(var(estimates, 1, corrected=false), 1)*(size(estimates, 1)-1)
    jn = computestat(MultiJackknifeSurrogates(stat, nmulti), csinput, csinput)
    @test_approx_eq jn.trueval trueval
    @test_approx_eq_eps jackknife_bias(jn) (size(estimates, 1)-1)*(squeeze(mean(estimates, 1), 1) - trueval) sqrt(eps())
    @test_approx_eq jackknife_var(jn) jnvar

    # Test Bootstrap
    estimates = zeros(eltype(trueval), size(csinput, 2), size(csinput, 2), size(bsindices, 2))
    for i = 1:size(bsindices, 2)
        estimates[:, :, i] = computestat(stat, csinput[bsindices[:, i], :], csinput[bsindices[:, i], :])
    end
    bs = computestat(Bootstrap(stat, bsindices), csinput, csinput)
    @test_approx_eq bs estimates

    # Test Bootstrap{GroupMean}
    bs = computestat(Bootstrap(gmstat, bsindices), csinput, csinput)
    @test_approx_eq bs squeeze([estimates[1, 2, :]; estimates[2, 3, :]; (estimates[1, 2, :] + estimates[2, 3, :])/2], 2)

    # Test Permutation (need a better test here)
    estimates = zeros(eltype(trueval), size(csinput, 2), size(csinput, 2), size(permindices, 2))
    for i = 1:size(permindices, 2)
        estimates[:, :, i] = computestat(stat, csinput[permindices[:, i], :], csinput)
    end
    perms = computestat(Permutation(stat, permindices), csinput, csinput)
    @test_approx_eq perms estimates

    # Test Permutation{GroupMean}
    bs = computestat(Permutation(gmstat, permindices), csinput, csinput)
    @test_approx_eq bs squeeze([estimates[1, 2, :]; estimates[2, 3, :]; (estimates[1, 2, :] + estimates[2, 3, :])/2], 2)
end

# Test nd
offsets = reshape(rand(10)*2pi, 1, 10)
snr = 5*reshape(rand(10), 1, 10)
inputs = rand(100, 10, 1, 7).*exp(im*(offsets .+ randn(100, 10, 1, 7)./snr))

ngroups = 10
groups = Array(Vector{Tuple{Int, Int}}, ngroups)
for igroup = 1:ngroups
    pairs = Array(Tuple{Int, Int}, rand(1:10))
    for ipair = 1:length(pairs)
        ch1 = rand(1:size(inputs, 2))
        ch2 = rand(1:size(inputs, 2)-1)
        ch2 >= ch1 && (ch2 += 1)
        pairs[ipair] = (ch1, ch2)
    end
    groups[igroup] = pairs
end

nbootstraps = 10

for stat in [Coherence, Coherency, MeanPhaseDiff, PLV, PPC, PLI, PLI2Unbiased, WPLI, WPLI2Debiased, JammalamadakaR, JuppMardiaR]
    # True statistic
    tv = mapslices(x->computestat(stat(), x), inputs, (1, 2))
    @test_approx_eq computestat(stat(), inputs) tv
    @test_approx_eq computestat_parallel(stat(), inputs) tv

    # GroupMean
    gmstat = GroupMean(stat(), size(inputs, 2), groups)
    tgm = mapslices(x->computestat(gmstat, x), inputs, (1, 2))
    @test_approx_eq computestat(gmstat, inputs) tgm
    @test_approx_eq computestat_parallel(gmstat, inputs) tgm

    # JackknifeSurrogates
    jns = computestat(JackknifeSurrogates(stat()), inputs)
    @test_approx_eq jns.trueval tv
    @test_approx_eq jackknife_var(jns) mapslices(x->jackknife_var(computestat(JackknifeSurrogates(stat()), x)), inputs, (1, 2))

    # Jackknife
    jn = computestat(Jackknife(stat()), inputs)
    @test_approx_eq jn.trueval tv
    @test_approx_eq jn.var mapslices(x->computestat(Jackknife(stat()), x).var, inputs, (1, 2))

    # Jackknife{GroupMean}
    jn = computestat(Jackknife(gmstat), inputs)
    @test_approx_eq jn.trueval mapslices(x->computestat(Jackknife(gmstat), x).trueval, inputs, (1, 2))
    @test_approx_eq jn.var mapslices(x->computestat(Jackknife(gmstat), x).var, inputs, (1, 2))

    # Bootstrap and Permutation
    for t in (Bootstrap, Permutation)
        bs = t(stat(), size(inputs, 1), nbootstraps)
        bstrue = zeros(eltype(tv), size(inputs, 2), size(inputs, 2), nbootstraps, size(inputs)[3:end]...)
        for i = 1:Base.trailingsize(inputs, 3)
            bstrue[:, :, :, i] = computestat(bs, inputs[:, :, i])
        end
        @test_approx_eq computestat(bs, inputs) bstrue
        @test_approx_eq computestat_parallel(bs, inputs) bstrue

        bs = t(gmstat, size(inputs, 1), nbootstraps)
        bstrue = zeros(eltype(tv), length(groups), nbootstraps, size(inputs)[3:end]...)
        for i = 1:Base.trailingsize(inputs, 3)
            bstrue[:, :, i] = computestat(bs, inputs[:, :, i])
        end
        @test_approx_eq computestat(bs, inputs) bstrue
        @test_approx_eq computestat_parallel(bs, inputs) bstrue
    end
end

# Test shift predictor
# x = 0:63
# signal = zeros(length(x), 1, 35)
# for i = 1:size(signal, 3)
#     signal[:, 1, i] = cospi(0.2*x+rand()*2)
# end
# c = multitaper([signal signal], Coherence())
# @test_approx_eq c ones(size(c))
# for lag = 1:5
#     t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], Coherence())
#     sp = multitaper([signal signal], ShiftPredictor(Coherence(), lag))
#     @test_approx_eq t sp
# end

# # Test jackknifed shift predictor
# for stat in (Coherence, PLV), lag = 1:5
#     t = multitaper([signal signal[:, :, circshift([1:size(signal, 3)], lag)]], JackknifeSurrogates(stat()))
#     sp = multitaper([signal signal], JackknifeSurrogates(ShiftPredictor(stat(), lag)))
#     @test_approx_eq t[1] sp[1]
#     (tbias, tvar) = jackknife_bias_var(t...)
#     (spbias, spvar) = jackknife_bias_var(sp...)
#     @test_approx_eq tbias spbias
#     @test_approx_eq tvar spvar
# end

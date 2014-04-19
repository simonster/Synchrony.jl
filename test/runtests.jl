tests = [
    "multitaper",
    "transform_stats",
    "wavelet"
]

println("Running tests:")

for t in tests
    println(" * $t.jl")
    include("$t.jl")
end

# TODO PLI, PLI2Unbiased, WPLI, WPLI2Debiased,
#      spiketriggeredspectrum, pfcoherence, pfplv, pfppc0, pfppc1,
#      pfppc2
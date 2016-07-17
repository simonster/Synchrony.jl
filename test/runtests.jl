tests = [
    "transform_stats",
    "wavelet",
    "point_field"
]

println("Running tests:")

for t in tests
    println(" * $t.jl")
    include("$t.jl")
end

# TODO point field tests
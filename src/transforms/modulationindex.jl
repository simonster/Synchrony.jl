#
# Entropy-based cross frequency coupling
#
# See Hurtado, J. M., Rubchinsky, L. L., & Sigvardt, K. A. (2004).
# Statistical Method for Detection of Phase-Locking Episodes in Neural
# Oscillations. Journal of Neurophysiology, 91(4), 1883–1898.
# doi:10.1152/jn.00853.2003
#
# Tort, A. B. L., Kramer, M. A., Thorn, C., Gibson, D. J., Kubota, Y.,
# Graybiel, A. M., & Kopell, N. J. (2008). Dynamic cross-frequency
# couplings of local field potential oscillations in rat striatum and
# hippocampus during performance of a T-maze task. Proceedings of the
# National Academy of Sciences, 105(51), 20517–20522.
# doi:10.1073/pnas.0810524105
immutable HurtadoModulationIndex <: PairwiseStatistic
    nbins::UInt8
    hmax::Float64

    HurtadoModulationIndex(nbins) = new(nbins, log(nbins))
end
HurtadoModulationIndex() = HurtadoModulationIndex(18)
Base.eltype{T<:Real}(::HurtadoModulationIndex, X::AbstractArray{Complex{T}}) = T

allocwork{T<:Real}(t::HurtadoModulationIndex, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(UInt8, size(X, 1), size(X, 2)), Array(Int32, t.nbins, nchannels(X)),
     Array(T, size(X, 1)), Array(Float64, t.nbins))

function binphases!{T<:Real}(phasebin::Matrix{UInt8}, ninbin::Matrix{Int32}, X::AbstractVecOrMat{Complex{T}})
    fill!(ninbin, UInt8(0))
    m = size(ninbin, 1)/2pi
    for j = 1:size(X, 2), i = 1:size(X, 1)
        @inbounds bin = phasebin[i, j] = ceil(UInt8, (angle(X[i, j])+pi)*m)
        @inbounds ninbin[bin, j] += 1
    end
end

function sumphases!{T<:Real}(meanamp::Vector{Float64}, phasebin::Matrix{UInt8}, amp::Vector{T}, iphase::Int)
    # Empty phase bins
    for i = 1:length(meanamp)
        @inbounds meanamp[i] = 0
    end

    # Compute sum of amplitudes for each phase bin
    for i = 1:size(phasebin, 1)
        @inbounds meanamp[phasebin[i, iphase]] += amp[i]
    end
end

function finish_mi!(meanamp::Vector{Float64}, ninbin::AbstractVecOrMat{Int32}, iphase::Int, hmax)
    # Normalize to mean amplitudes
    d = 0.
    for ibin = 1:length(meanamp)
        @inbounds d += (meanamp[ibin] /= ninbin[ibin, iphase])
    end
    d = inv(d)

    # Compute h
    h = 0.
    for ibin = 1:length(meanamp)
        @inbounds p = meanamp[ibin]*d
        h -= p*log(p)
    end

    # Compute modulation index
    (hmax - h)/hmax
end

# X is phase, Y is amplitude
function computestat!{T<:Real}(t::HurtadoModulationIndex, out::AbstractMatrix{T},
                               work::Tuple{Matrix{UInt8}, Matrix{Int32}, Vector{T}, Vector{Float64}},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    nbins = t.nbins
    phasebin, ninbin, amp, meanamp = work
    chkinput(out, X, Y)
    size(ninbin, 1) == length(meanamp) == nbins || throw(ArgumentError("invalid work object"))

    binphases!(phasebin, ninbin, X)

    # Compute phase-amplitude coupling
    for k = 1:size(Y, 2)
        # Compute amplitudes
        for i = 1:size(X, 1)
            @inbounds amp[i] = abs(Y[i, k])
        end

        for j = 1:size(X, 2)
            sumphases!(meanamp, phasebin, amp, j)
            out[j, k] = finish_mi!(meanamp, ninbin, j, t.hmax)
        end
    end

    out
end

allocwork{T<:Real}(t::AbstractJackknifeSurrogates{HurtadoModulationIndex}, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}=X) =
    (Array(UInt8, size(X, 1), size(X, 2)), Array(Int32, t.transform.nbins, nchannels(X)),
     Array(T, size(X, 1)), Array(Float64, t.transform.nbins), Array(Int32, t.transform.nbins), Array(Float64, t.transform.nbins))
function computestat!{T<:Real}(t::AbstractJackknifeSurrogates{HurtadoModulationIndex}, out::JackknifeSurrogatesOutput,
                               work::Tuple{Matrix{UInt8}, Matrix{Int32}, Vector{T}, Vector{Float64}, Vector{Int32}, Vector{Float64}},
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    trueval = out.trueval
    surrogates = out.surrogates
    nbins = t.transform.nbins
    phasebin, ninbin, amp, meanamp, ninbin2, meanamp2 = work
    chkinput(trueval, X, Y)
    size(ninbin, 1) == length(meanamp) == nbins || throw(ArgumentError("invalid work object"))
    ntrials(X) % jnn(t) == 0 || throw(DimensionMismatch("ntrials not evenly divisible by $(jnn(t))"))
    size(out.surrogates, 1) == div(ntrials(X), jnn(t)) || throw(DimensionMismatch("invalid output size"))

    binphases!(phasebin, ninbin, X)

    for k = 1:size(Y, 2)
        # Compute amplitudes
        for i = 1:size(X, 1)
            @inbounds amp[i] = abs(Y[i, k])
        end

        for j = 1:size(X, 2)
            sumphases!(meanamp, phasebin, amp, j)

            # Compute jackknifes
            for i = 1:size(surrogates, 1)
                for m = 1:nbins
                    @inbounds meanamp2[m] = meanamp[m]
                    @inbounds ninbin2[m] = ninbin[m, j]
                end
                for idel = (i-1)*jnn(t)+1:i*jnn(t)
                    @inbounds meanamp2[phasebin[idel, j]] -= amp[idel]
                    @inbounds ninbin2[phasebin[idel, j]] -= 1
                end
                surrogates[i, j, k] = finish_mi!(meanamp2, ninbin2, 1, t.transform.hmax)
            end

            # Compute true modulation index
            trueval[j, k] = finish_mi!(meanamp, ninbin, j, t.transform.hmax)
        end
    end

    out
end

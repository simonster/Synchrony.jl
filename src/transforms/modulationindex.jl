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
    nbins::Uint8
end
HurtadoModulationIndex() = HurtadoModulationIndex(18)
Base.eltype{T<:Real}(::HurtadoModulationIndex, X::AbstractArray{Complex{T}}) = T

allocwork{T<:Real}(t::HurtadoModulationIndex, X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}}) =
    (Array(Uint8, size(X, 1), size(X, 2)), Array(Int, t.nbins, nchannels(X)),
     Array(T, size(Y, 1), size(Y, 2)), Array(Float64, t.nbins))
function computestat!{T<:Real}(t::HurtadoModulationIndex, out::AbstractMatrix{T},
                               work::@compat(Tuple{Matrix{Uint8}, Matrix{Int}, Matrix{T}, Vector{Float64}}),
                               X::AbstractVecOrMat{Complex{T}}, Y::AbstractVecOrMat{Complex{T}})
    nbins = t.nbins
    phasebin = work[1]
    ninbin = work[2]
    amp = work[3]
    meanamp = work[4]
    chkinput(out, X, Y)

    # Bin phases
    fill!(ninbin, UInt8(0))
    m = nbins/2pi
    for j = 1:size(X, 2), i = 1:size(X, 1)
        @inbounds bin = phasebin[i, j] = ceil(Uint8, (angle(X[i, j])+pi)*m)
        @inbounds ninbin[bin, j] += 1
    end

    hmax = log(nbins)
    # Compute phase-amplitude coupling
    for k = 1:size(Y, 2)
        # Compute amplitudes
        for i = 1:size(X, 1)
            @inbounds amp[i] = abs(Y[i, k])
        end

        for j = 1:size(X, 2)
            fill!(meanamp, zero(T))

            # Compute sum of amplitudes for each phase bin
            @simd for i = 1:size(X, 1)
                @inbounds meanamp[phasebin[i]] += amp[i]
            end

            # Normalize to mean amplitudes
            d = zero(T)
            for ibin = 1:nbins
                d += meanamp[ibin] /= ninbin[ibin]
            end
            d = inv(d)

            # Compute h
            h = zero(T)
            for ibin = 1:nbins
                p = meanamp[ibin]*d
                h -= p*log(p)
            end

            # Compute modulation index
            out[j, k] = (hmax - h)/hmax
        end
    end

    out
end

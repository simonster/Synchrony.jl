#
# Entropy-based cross frequency coupling
#
# See:
# Hurtado, J. M., Rubchinsky, L. L., & Sigvardt, K. A. (2004).
# Statistical Method for Detection of Phase-Locking Episodes in Neural
# Oscillations. Journal of Neurophysiology, 91(4), 1883–1898.
# doi:10.1152/jn.00853.2003
# Tort, A. B. L., Kramer, M. A., Thorn, C., Gibson, D. J., Kubota, Y.,
# Graybiel, A. M., & Kopell, N. J. (2008). Dynamic cross-frequency
# couplings of local field potential oscillations in rat striatum and
# hippocampus during performance of a T-maze task. Proceedings of the
# National Academy of Sciences, 105(51), 20517–20522.
# doi:10.1073/pnas.0810524105
immutable HurtadoModulationIndex <: RealPairwiseStatistic
    nbins::Uint8
end
HurtadoModulationIndex() = HurtadoModulationIndex(18)

allocwork{T<:Real}(t::HurtadoModulationIndex, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}}) =
    (Array(Uint8, size(X, 1)), Array(Int, t.nbins, size(X, 1)),
     Array(Float64, t.nbins, size(X, 1), size(Y, 1)),
     Array(Float64, t.nbins))
function computestat!{T<:Real}(t::HurtadoModulationIndex, out::AbstractMatrix{T},
                               work::(Array{Uint8, 1}, Array{Int, 2}, Array{Float64, 3},
                                      Vector{Float64}),
                               X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix{Complex{T}})
    nbins = t.nbins
    phasebin = work[1]
    ninbin = work[2]
    ampsum = work[3]
    meanamp = work[4]
    fill!(ninbin, 0)
    fill!(ampsum, zero(T))

    m = nbins/2pi
    for j = 1:size(X, 2)
        for i = 1:size(X, 1)
            @inbounds bin = phasebin[i] = ceil(Uint8, (angle(X[i, j])+pi)*m)
            @inbounds ninbin[bin, i] += 1
        end
        
        for iampchan = 1:size(Y, 1)
            @inbounds amp = abs(Y[iampchan, j])
            for iphasechan = 1:size(X, 1)
                @inbounds ampsum[phasebin[iphasechan], iphasechan, iampchan] += amp
            end
        end
    end

    hmax = log(float64(nbins))
    for iampchan = 1:size(Y, 1), iphasechan = 1:size(X, 1)
        d = zero(T)/1+zero(T)/1
        for ibin = 1:nbins
            d += meanamp[ibin] = ampsum[ibin, iphasechan, iampchan]/ninbin[ibin, iphasechan]
        end
        d = inv(d)

        h = zero(T)/1+zero(T)/1
        for ibin = 1:nbins
            p = meanamp[ibin]*d
            h -= p*log(p)
        end

        out[iphasechan, iampchan] = (hmax - h)/hmax
    end

    out
end
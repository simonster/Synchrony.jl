# FrequencyDomainAnalysis.jl
# Tools for spectral density estimation and analysis of phase relationships
# between sets of signals.

# Copyright (C) 2013   Simon Kornblith

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

export multitaper, psd, xspec, coherence

# Perform tapered FFT of continuous signals
# A is samples x channels x trials
function multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                             stats::(TransformStatistic, TransformStatistic...);
                             tapers::Union(Vector, Matrix)=dpss(size(A, 1), 4), nfft::Int=nextpow2(size(A, 1)),
                             fs::Real=1.0, freqrange::Range1{Int}=1:(nfft >> 1 + 1))
    n = size(A, 1)
    nout = nfft >> 1 + 1
    ntapers = size(tapers, 2)
    nchannels = size(A, 2)
    multiplier = sqrt(2/fs)

    for stat in stats
        init(stat, length(freqrange), nchannels, ntapers)
    end

    dtype = outputtype(T)
    fftin = zeros(dtype, nfft, size(A, 2))
    fftout = Array(Complex{dtype}, nout, size(A, 2))
    #fftview = sub(fftout, freqrange, 1:size(A,2))
    fftview = zeros(Complex{dtype}, length(freqrange), size(A, 2))

    p = FFTW.Plan(fftin, fftout, 1, FFTW.ESTIMATE, FFTW.NO_TIMELIMIT)

    for j = 1:size(A, 3), i = 1:ntapers
        @inbounds for k = 1:nchannels, l = 1:n
            fftin[l, k] = A[l, k, j]*tapers[l, i]
        end

        FFTW.execute(dtype, p.plan)
        copyscalefft!(fftview, fftout, freqrange, multiplier, nfft)

        for stat in stats
            accumulate(stat, fftview, i)
        end
    end

    [finish(stat) for stat in stats]
end

# Calling with types instead of instances
multitaper{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                    stat::(DataType, DataType...); kw...) =
    multitaper(A, tuple([x{outputtype(T)}() for x in stat]...); kw...)

# Calling with a single type or instance
multitaper{T<:Real,S<:TransformStatistic}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                                          stat::Union(S, Type{S}); kw...) = multitaper(A, (stat,); kw...)[1]

#
# Convenience functions
#

psd{T<:Real}(A::AbstractArray{T}; kw...) =
    multitaper(A, (PowerSpectrum{T}(),); kw...)[1]
xspec{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (CrossSpectrum{T}(),); kw...)[1]
coherence{T<:Real}(A::Vector{T}, B::Vector{T}; kw...) =
    multitaper(hcat(A, B), (Coherence{T}(),); kw...)[1]

# Copy FFT from one buffer to another, including only frequencies in
# frange, simultaneously multiplying by multiplier, and correcting
# edges according to nfft
function copyscalefft!(fftview, fftout, frange, multiplier, nfft)
    divisor = 1/sqrt(2)
    @inbounds for k = 1:size(fftout, 2)
        for l = 1:length(frange)
            fftview[l, k] = fftout[frange[l], k]*multiplier
        end

        # Correct edges of FFT so that all power is in the same units
        # This is necessary because the FFT divides power associated
        # with real signals between positive and negative frequencies,
        # except for power at 0 and FMAX. We don't compute the
        # negative frequencies, but we do need to scale the FFT so
        # that the power is correct.
        if frange[1] == 1
            fftview[1, k] *= divisor
        end
        if frange[end] == size(fftout, 1) && iseven(nfft)
            fftview[end, k] *= divisor
        end
    end
end

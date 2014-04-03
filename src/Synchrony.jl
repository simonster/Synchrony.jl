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

module Synchrony
using DSP, NumericExtensions, ArrayViews

export frequencies

# Get next fast FFT size for a given signal length
nextfastfft(n) = nextprod([2, 3, 5, 7], n)

# Compute frequencies based on length of padded FFT
function frequencies(nfft::Int, fs::Real=1.0)
    freq = (0:div(nfft, 2))*(fs/nfft)
    (freq, 1:length(freq))
end

# Compute frequencies based on length of padded FFT, with limits
function frequencies(nfft::Int, fs::Real, fmin::Real, fmax::Real=Inf)
    (freq,) = frequencies(nfft, fs)
    freqrange = searchsortedfirst(freq, fmin):(fmax == Inf ? length(freq) : searchsortedlast(freq, fmax))
    (freq[freqrange], freqrange)
end

# Compute frequencies based on data length, assuming padding to nextfastfft
frequencies{T<:Real}(A::Union(AbstractVector{T}, AbstractMatrix{T}, AbstractArray{T,3}),
                     fs::Real=1.0, fmin::Real=0.0, fmax::Real=Inf) =
    frequencies(nextfastfft(size(A, 1)), fs, fmin, fmax)

# Get the equivalent complex type for a given type
complextype{T<:Complex}(::Type{T}) = T
complextype{T}(::Type{T}) = Complex{T}

# Get preferred output type for a given input type
outputtype{T<:FloatingPoint}(::Type{T}) = T
outputtype(::Union(Type{Int8}, Type{Uint8}, Type{Int16}, Type{Uint16})) = Float64
outputtype{T<:Real}(::Type{T}) = Float64

include("transform_stats.jl")
include("multitaper.jl")
include("point_field.jl")
include("wavelet.jl")
end
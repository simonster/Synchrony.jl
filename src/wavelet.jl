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

import Base: getindex, size, ndims, convert
export MorletWavelet, wavebases

#
# Mother wavelets, which are convolved with the signal in frequency space
#
abstract MotherWavelet
abstract WaveletBases{T} <: AbstractArray{T,2}

ndims(c::WaveletBases) = 2
size(c::WaveletBases, x::Int) =
    x == 1 ? c.n : x == 2 ? length(c.w.foi) : 0
size(c::WaveletBases) = (c.n, length(c.w.foi))
convert{T}(::Union(Type{Array{T}}, Type{Array{T,2}}), c::WaveletBases{T}) =
    [c[j, k] for j = 1:c.n, k = 1:length(c.w.foi)]

immutable MorletWavelet{T} <:MotherWavelet
    k0::T
    foi::Vector{T}
    fourierfactor::T
end
MorletWavelet{T<:Real}(foi::Vector{T}, k0::Real=5.0) =
    MorletWavelet(convert(T, k0), foi, convert(T, (4pi)/(k0 + sqrt(2 + k0^2))))
function wavebases(w::MorletWavelet, n::Int, fs::Real=1)
    scales = 2pi * fs./(n * w.foi * w.fourierfactor)
    norms = sqrt(scales / sqrt(pi) * n)
    nmax = div(n, 2) + 1
    MorletWaveletBases(n, nmax, w, scales, norms)
end

immutable MorletWaveletBases{T} <: WaveletBases{T}
    n::Int
    nmax::Int
    w::MorletWavelet{T}
    scales::Vector{T}
    norms::Vector{T}
end
getindex(c::MorletWaveletBases, j::Int, k::Int) =
    (j > 1 && j <= c.nmax) * c.norms[k] * exp(-abs2(c.scales[k] * (j - 1) - c.w.k0)*0.5)


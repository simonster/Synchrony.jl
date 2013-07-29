module Multitaper
using NumericExtensions

export dpss, psd, xspec, coherence, mtfft

# Compute discrete prolate spheroid sequences (Slepian tapers)
function dpss(n::Int, nw::Real, ntapers::Int=iceil(2*nw)-1)
	# Construct symmetric tridiagonal matrix
	# See Gruenbacher, D. M., & Hummels, D. R. (1994). A simple algorithm
	# for generating discrete prolate spheroidal sequences. IEEE
	# Transactions on Signal Processing, 42(11), 3276-3278.
	i1 = 0:(n-1)
	i2 = 1:(n-1)
	mat = SymTridiagonal(cos(2pi*nw/n)*((n - 1)/2 - i1).^2, 0.5.*(i2*n - i2.^2))

	# Get tapers
	ev = eigvals(mat, n-ntapers+1, n)
	v = fliplr(eigvecs(mat, ev)[1])

	# Polarity convention is that first lobe of DPSS is positive
	sgn = ones(size(v, 2))
	sgn[2:2:end] = sign(v[2, 2:2:end])
	scale!(v, sgn)
end

#
# Front-ends with automatic memory allocation
#
for (fn, invars, allocdims, outtype) in ((:psd, (:A,), (), (:(eltype(A)),)),
		                                 (:xspec, (:A, :B), (2,), (:(complextype(eltype(A))), :(eltype(A)), :(eltype(A)))))
	outvars = [symbol("out$i") for i = 1:length(outtype)]
	@eval begin
		function $fn{T<:FFTW.fftwNumber}($([:($x::AbstractArray{T}) for x in invars]...);
			                             tapers::Matrix=dpss(size($(invars[1]), 1), 4), fs::Real=1.0,
			                             pad::Union(Bool, Int)=true)
			npadded = getpadding(size(A, 1), pad)
			outsz = tuple(div(npadded, 2)+1, size(A)[2:end]...)
			tmp = zeros(complextype(eltype(A)), npadded, prod(size(A)[2:end])::Int, $(allocdims...))
			$(Expr(:block, [:($(outvars[i]) = zeros($(outtype[i]), outsz)) for i = 1:length(outtype)]...))
			$(Expr(:call, symbol(string(fn, "!")), outvars..., :tmp, invars..., :tapers, :fs))
		end
	end
end

#
# Basic functionality, for single series and pairs
#
# Estimate power spectral density
function psd!(out, tmp, A, tapers, fs)
	n = size(A, 1)
	nfft = size(tmp, 1)
	nout = size(out, 1)
	zerorg = n+1:nfft
	ntapers = size(tapers, 2)
	perform_fft! = plan_fft!(tmp, 1)

	for i = 1:ntapers
		for j = 1:div(length(A), n)
			Aoff = (j-1)*n
			for k = 1:n
				tmp[k, j] = A[Aoff+k].*tapers[k, i]
			end
			tmp[zerorg, j] = zero(eltype(tmp))
		end
		
		perform_fft!(tmp)

		for j = 1:div(length(out), nout)
			outoff = (j-1)*nout
			for k = 1:nout
				out[outoff+k] += abs2(tmp[k, j])
			end
		end
	end
	scalespectrum!(out, nfft, fs*ntapers)
end

# Estimate cross-spectrum
function xspec!(sXY, sXX, sYY, tmp, A, B, tapers, fs)
	if size(A) != size(B)
		throw(Base.DimensionMismatch("A and B must be the same size"))
	end

	n = size(A, 1)
	nfft = size(tmp, 1)
	nout = size(sXY, 1)
	zerorg = n+1:nfft
	ntapers = size(tapers, 2)
	perform_fft! = plan_fft!(tmp, 1)

	for i = 1:ntapers
		for j = 1:div(length(A), n)
			Aoff = (j-1)*n
			for k = 1:n
				tmp[k, j, 1] = A[Aoff+k].*tapers[k, i]
				tmp[k, j, 2] = B[Aoff+k].*tapers[k, i]
			end
			tmp[zerorg, j, :] = zero(eltype(tmp))
		end
		
		perform_fft!(tmp)

		for j = 1:div(length(sXY), nout)
			outoff = (j-1)*nout
			for k = 1:nout
				outind = outoff+k
				sXY[outind] += dot(tmp[k, j, 1], tmp[k, j, 2])
				sXX[outind] += abs2(tmp[k, j, 1])
				sYY[outind] += abs2(tmp[k, j, 2])
			end
		end
	end
	(scalespectrum!(sXY, nfft, fs*ntapers),
	 scalespectrum!(sXX, nfft, fs*ntapers),
	 scalespectrum!(sYY, nfft, fs*ntapers))
end

# Estimate coherence
function coherence{T<:FFTW.fftwNumber}(A::AbstractArray{T}, B::AbstractArray{T},
	                                   tapers::Matrix=dpss(size(data, 1), 4);
	                                   pad::Union(Bool, Int)=true)
	sXY, sXX, sYY = xspec(A, B; tapers::Matrix=tapers, pad=pad)
	for i = 1:length(sXY)
		sXX[i] = abs(sXY[i])/sqrt(sXX[i]*sYY[i])
	end
	sXX
end

#
# Functionality for multiple channels
#
# A is time x trials x channels
function xspec{T<:FFTW.fftwNumber}(A::AbstractArray{T, 3}, tapers::Matrix=dpss(size(data, 1), 4);
	                               pad::Union(Bool, Int)=true, trialavg::Bool=true)
	npadded = getpadding(size(A, 1), pad)
	X = mtfft(A, tapers)
	combs = [(i, j) for i = 1:size(A, 3)-1, j = i+1:size(A, 3)]
	s = scale!(sqsum(X, 4), 1/size(A, 4))
	if trialavg
		s = scale!(sqsum(s, 2), 1/size(s, 2))
		X = sum(X, 4)
		xs = zeros(size(X, 1), 1, length(combs))
		for k = 1:length(combs)
			ch1, ch2 = combs[k]
			for j = 1:size(X,2), i = 1:size(X, 1)
				xs[i, 1, k] += dot(X[i, j, ch1, 1], X[i, j, ch2, 1])
			end
		end
		scale!(xs, 1./(size(X, 2)*size(X, 4)))
	else
		xs = zeros(size(X, 1), size(X, 2), length(combs))
		for l = 1:size(X,4), k = 1:length(combs)
			ch1, ch2 = combs[k]
			for j = 1:size(X,2), i = 1:size(X, 1)
				xs[i, j, k] += dot(X[i, j, ch1, l], X[i, j, ch2, l])
			end
		end
		scale!(xs, 1/size(X, 4))
	end
	(xs, s, combs)
end

function coherence{T<:FFTW.fftwNumber}(A::AbstractArray{T, 3}, tapers::Matrix=dpss(size(data, 1), 4);
	                                   pad::Union(Bool, Int)=true, trialavg::Bool=true)
	(xs, s, combs) = coherence(A, tapers::Matrix=tapers, pad=pad, trialavg=trialavg)
	for k = 1:length(combs), j = 1:size(A, 2), i = 1:size(A, 1)
		ch1, ch2 = combs[k]
		xs[i, j, k] = abs(xs[i, j, k])/sqrt(s[i, j, ch1]*s[i, j, ch2])
	end
	(xs, combs)
end

#
# Helper functions
#
# Perform tapered FFT
function mtfft{T<:FFTW.fftwNumber,N}(A::AbstractArray{T,N}, tapers::Matrix)
	sz = size(A)
	l = length(A)

	X = Array(eltype(A), sz..., size(tapers, 2))::Array{T,N+1}
	for i = 1:size(tapers, 2)
		t = (i-1)*l
		for j = 0:n:length(A)-2, k = 1:size(tapers, 1)
			X[j+k+l] = A[j+k+l].*tapers[k, i]
		end
		X[j+l+(n+1:nout)] = zero(eltype(y))
	end
	rfft(X, 1)
end

getpadding(n::Int, padparam::Bool) = padparam ? nextpow2(n) : n
getpadding(n::Int, padparam::Int) = padparam

function scalespectrum!(spectrum::Union(Vector, Matrix), n::Int, divisor::Real)
	scale!(spectrum, 2/divisor)
	spectrum[1, :] /= 2
	if iseven(n)
		spectrum[end, :] /= 2
	end
	spectrum
end

# Get the equivalent complex type for a given type
complextype{T<:Complex}(::Type{T}) = T
complextype{T}(::Type{T}) = Complex{T}
end
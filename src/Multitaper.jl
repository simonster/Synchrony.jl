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
			nfft = getpadding(size(A, 1), pad)
			outsz = tuple(div(nfft, 2)+1, size(A)[2:end]...)
			tmp = zeros(complextype(eltype(A)), nfft, prod(size(A)[2:end])::Int, $(allocdims...))
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
function coherence{T<:FFTW.fftwNumber}(A::AbstractArray{T}, B::AbstractArray{T};
	                                   tapers::Matrix=dpss(size(A, 1), 4),
	                                   pad::Union(Bool, Int)=true, fs::Real=1.0)
	sXY, sXX, sYY = xspec(A, B; tapers=tapers, pad=pad, fs=fs)
	for i = 1:length(sXY)
		sXY[i] = sXY[i]/sqrt(sXX[i]*sYY[i])
	end
	sXY
end

#
# Functionality for multiple channels
#
# A is time x trials x channels
function xspec{T<:FFTW.fftwNumber}(A::AbstractArray{T, 3}; tapers::Matrix=dpss(size(A, 1), 4),
	                               pad::Union(Bool, Int)=true, fs::Real=1.0)
	nfft = getpadding(size(A, 1), pad)
	X = mtrfft(A, tapers, nfft)
	scalespectrum!(X, nfft, fs*size(X, 4), true)

	combs = Array((Int, Int), binomial(size(A, 3), 2))
	k = 0
	for i = 1:size(A, 3)-1, j = i+1:size(A, 3)
		combs[k += 1] = (i, j)
	end

	s = squeeze(sqsum(X, 4), 4)
	xs = zeros(complextype(eltype(A)), size(X, 1), size(X, 2), length(combs))
	for l = 1:size(X, 4), k = 1:length(combs)
		ch1, ch2 = combs[k]
		for j = 1:size(X, 2), i = 1:size(X, 1)
			xs[i, j, k] += dot(X[i, j, ch1, l], X[i, j, ch2, l])
		end
	end

	(xs, s, combs)
end

function coherence{T<:FFTW.fftwNumber}(A::AbstractArray{T, 3}; tapers::Matrix=dpss(size(A, 1), 4),
	                                   pad::Union(Bool, Int)=true, fs::Real=1.0,
	                                   trialavg::Bool=true)
	(xs, s, combs) = xspec(A; tapers=tapers, pad=pad, fs=fs)
	if trialavg
		xs = mean(xs, 2)
	end
	for k = 1:length(combs), j = 1:size(xs, 2), i = 1:size(xs, 1)
		ch1, ch2 = combs[k]
		xs[i, j, k] = xs[i, j, k]/sqrt(s[i, j, ch1]*s[i, j, ch2])
	end
	(xs, s, combs)
end

#
# Helper functions
#
# Perform tapered FFT
function mtrfft{T<:FFTW.fftwNumber,N}(A::AbstractArray{T,N}, tapers::Matrix, nfft::Int=size(A, 1))
	n = size(A, 1)
	sz = size(A)

	X = Array(eltype(A), nfft, sz[2:end]..., size(tapers, 2))::Array{T,N+1}
	Xstride = stride(X, N+1)
	for i = 1:size(tapers, 2), j = 1:div(length(A), n)
		Aoff = (j-1)*n
		Xoff = (j-1)*nfft+Xstride*(i-1)
		for k = 1:n
			X[Xoff+k] = A[Aoff+k].*tapers[k, i]
		end
		X[Xoff+(n+1:nfft)] = zero(eltype(X))
	end
	rfft(X, 1)
end

getpadding(n::Int, padparam::Bool) = padparam ? nextpow2(n) : n
getpadding(n::Int, padparam::Int) = padparam

# Scale real part of spectrum to satisfy Parseval's theorem
# If sq is true, scale by sqrt of values (for FFTs)
function scalespectrum!(spectrum::AbstractArray, n::Int, divisor::Real, sq::Bool=false)
	d = 2/divisor
	if sq; d = sqrt(d); end
	scale!(spectrum, d)

	s = size(spectrum, 1)
	for i = 1:div(length(spectrum), s)
		spectrum[(i-1)*s+1] /= sq ? sqrt(2) : 2
		if iseven(n)
			spectrum[i*s] /= sq ? sqrt(2) : 2
		end
	end
	spectrum
end

# Get the equivalent complex type for a given type
complextype{T<:Complex}(::Type{T}) = T
complextype{T}(::Type{T}) = Complex{T}
end
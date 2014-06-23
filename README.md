# Analysis of synchronous signals

[![Build Status](https://travis-ci.org/simonster/Synchrony.jl.png?branch=master)](https://travis-ci.org/simonster/Synchrony.jl)

This package implements efficient multitaper and continuous wavelet transforms, along with the following transform statistics most of which operate on pairs of signals:

- Power spectral density (`PowerSpectrum`)
- Power spectral density variance (`PowerSpectrumVariance`)
- Cross spectrum (`CrossSpectrum`)
- Coherence (`Coherence` for the absolute value, `Coherency` for the complex value)
- Phase locking value, a.k.a. the mean resultant vector length or R̄ (`PLV`)
- Pairwise phase consistency, a.k.a. the unbiased estimator of R̄^2 (`PPC`)
- Phase lag index (`PLI`)
- Unbiased squared phase lang index (`PLI2Unbiased`)
- Weighted phase lag index (`WPLI`)
- Debiased squared weighted phase lag index (`WPLI2Debiased`)
- Jammalamadaka circular correlation coefficient (`JCircularCorrelation`)
- Jupp-Mardia squared circular correlation coefficient (`JMCircularCorrelation`)
- Hurtado et al. modulation index (phase-amplitude coupling) (`HurtadoModulationIndex`)

Additionally, the following point-field measures are implemented:

- Point-field coherence (`pfcoherence`)
- Point-field PLV (`pfplv`)
- Point-field PPC, variants 0, 1, and 2 (`pfppc0`, `pfppc1`, `pfppc2`)

And the following point-point measures:

- Point-point cross correlation (`pfxcorr`)

All measures except for the point-field measures have corresponding unit tests. Documentation is forthcoming.

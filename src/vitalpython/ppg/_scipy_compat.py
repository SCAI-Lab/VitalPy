"""`ricker` and `cwt` as SciPy implemented them up to 1.14.

SciPy deprecated both in 1.12 and removed them in 1.15, so on any modern SciPy
`import ...spectral_features` died at module import time with
`AttributeError: module 'scipy.signal' has no attribute 'ricker'`.

The alternative -- pinning scipy<1.15 -- would hold the whole stack back
(torch, numba, scikit-learn all want a recent SciPy), and the wavelet features
feed published numbers, so a differently-normalised replacement wavelet is not
acceptable either. These are verbatim ports of the removed SciPy sources, so
the feature values are unchanged. SciPy's own versions are preferred when
present, which keeps an older pinned environment bit-identical.
"""
import numpy as np
import scipy.signal

__all__ = ["ricker", "cwt"]


def _ricker(points, a):
    """Mexican-hat wavelet. Port of scipy.signal.ricker (removed in 1.15)."""
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec ** 2
    mod = 1 - xsq / wsq
    gauss = np.exp(-xsq / (2 * wsq))
    return A * mod * gauss


def _cwt(data, wavelet, widths, dtype=None, **kwargs):
    """Continuous wavelet transform. Port of scipy.signal.cwt (removed in 1.15)."""
    if dtype is None:
        if np.asarray(wavelet(1, widths[0], **kwargs)).dtype.char in "FDG":
            dtype = np.complex128
        else:
            dtype = np.float64

    output = np.empty((len(widths), len(data)), dtype=dtype)
    for ind, width in enumerate(widths):
        N = np.min([10 * width, len(data)])
        wavelet_data = np.conj(wavelet(N, width, **kwargs)[::-1])
        output[ind] = scipy.signal.convolve(data, wavelet_data, mode="same")
    return output


ricker = getattr(scipy.signal, "ricker", _ricker)
cwt = getattr(scipy.signal, "cwt", _cwt)

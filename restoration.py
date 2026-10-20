"""Image degradation and restoration helpers (motion blur, Gaussian noise,
inverse and Wiener filtering), all done in the 2-D DFT domain.

Data model (circular convolution, so it is exact in the DFT domain):
    g = h (*) f + n      <=>      G(u,v) = H(u,v) F(u,v) + N(u,v)
f: original image, h: motion-blur PSF, n: i.i.d. Gaussian noise N(0, sigma^2),
independent of f.
"""
import numpy as np
import scipy.io


def load_images(path="img_restoration.mat"):
    """Return the two test images as float64 arrays on the 0-255 scale."""
    d = scipy.io.loadmat(path)
    return {k: d[k].astype(np.float64) for k in ("I1", "I2")}


# ---------------------------------------------------------------- degradation

def motion_psf(length, angle_deg, shape):
    """Linear motion-blur PSF of `length` pixels at `angle_deg`, placed on a
    grid of size `shape` with its centre at index (0, 0) (wrapped around), so
    that fft2(psf) is the transfer function H without a phase shift.
    Sums to 1 (blur preserves mean brightness)."""
    M, N = shape
    psf = np.zeros(shape)
    theta = np.deg2rad(angle_deg)
    # Sample the line segment densely and accumulate into the nearest pixels.
    t = np.linspace(-(length - 1) / 2, (length - 1) / 2, 20 * max(length, 1))
    rows = np.round(-t * np.sin(theta)).astype(int) % M
    cols = np.round(t * np.cos(theta)).astype(int) % N
    np.add.at(psf, (rows, cols), 1.0)
    return psf / psf.sum()


def transfer_function(psf):
    return np.fft.fft2(psf)


def blur(f, H):
    """Circular convolution with the PSF, done as a product in the DFT domain."""
    return np.real(np.fft.ifft2(np.fft.fft2(f) * H))


def add_noise(f, sigma, rng):
    return f + rng.normal(0.0, sigma, f.shape)


# ---------------------------------------------------------------- restoration

def inverse_filter(g, H, eps=0.0):
    """Inverse filter F_hat = G / H.  Frequencies where |H| <= eps are set to
    zero (truncated / pseudo-inverse) to avoid dividing by ~0."""
    G = np.fft.fft2(g)
    R = np.zeros_like(H)
    keep = np.abs(H) > eps
    R[keep] = 1.0 / H[keep]
    return np.real(np.fft.ifft2(R * G))


def wiener_filter(g, H, nsr):
    """Wiener filter F_hat = H* / (|H|^2 + S_n/S_f) G.

    `nsr` is the noise-to-signal power ratio S_n(u,v)/S_f(u,v): either a scalar
    (constant-K approximation) or an array of the image size (per frequency)."""
    G = np.fft.fft2(g)
    R = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return np.real(np.fft.ifft2(R * G))


def oracle_nsr(f, sigma):
    """Per-frequency NSR using the true image: S_f = |F|^2 / (MN) (periodogram)
    and S_n = sigma^2 (white noise).  Upper bound on Wiener performance."""
    Sf = np.abs(np.fft.fft2(f)) ** 2 / f.size
    return sigma ** 2 / np.maximum(Sf, 1e-12)


# -------------------------------------------------------------------- metrics

def mse(f, f_hat):
    return np.mean((f - f_hat) ** 2)


def psnr(f, f_hat, peak=255.0):
    return 10 * np.log10(peak ** 2 / mse(f, f_hat))

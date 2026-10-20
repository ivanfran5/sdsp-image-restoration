"""Image degradation and restoration: motion blur, Gaussian noise,
inverse filter and Wiener filter.

Data model:   g = h * f + n      (in the DFT domain:  G = H F + N)
  f : original image
  h : motion-blur kernel (PSF), H = fft2(h)
  n : white Gaussian noise with standard deviation sigma
  g : degraded image
The convolution is circular, so G = H F holds exactly for the DFT.
"""
import numpy as np
import scipy.io


def load_images(path):
    """Load I1 and I2 from the .mat file as float arrays (values 0-255)."""
    data = scipy.io.loadmat(path)
    return {"I1": data["I1"].astype(float), "I2": data["I2"].astype(float)}


# ------------------------------------------------------------------
# Degradation
# ------------------------------------------------------------------

def motion_psf(length, angle_deg, shape):
    """Motion-blur kernel: a straight line of `length` pixels at `angle_deg`.

    The kernel is stored in an array of the same size as the image, with
    its centre at pixel (0, 0). Negative indices wrap around to the other
    side of the array (Python does that automatically), so the line is
    centred and the blur does not shift the image.
    """
    psf = np.zeros(shape)
    angle = np.deg2rad(angle_deg)
    for k in range(length):
        t = k - (length - 1) / 2               # position along the line
        row = int(round(-t * np.sin(angle)))   # minus: rows go downwards
        col = int(round(t * np.cos(angle)))
        psf[row, col] += 1
    return psf / psf.sum()                     # sum 1: keeps the brightness


def blur(f, H):
    """Blur image f: multiply its spectrum by H."""
    return np.real(np.fft.ifft2(np.fft.fft2(f) * H))


def add_noise(f, sigma, rng):
    """Add white Gaussian noise with standard deviation sigma."""
    return f + sigma * rng.standard_normal(f.shape)


# ------------------------------------------------------------------
# Restoration filters
# ------------------------------------------------------------------

def inverse_filter(g, H, eps):
    """Inverse filter: F_hat = G / H.

    Where |H| <= eps we cannot divide safely (division by almost zero
    blows up the noise), so those frequencies are set to zero.
    """
    G = np.fft.fft2(g)
    F_hat = np.zeros_like(G)
    ok = np.abs(H) > eps
    F_hat[ok] = G[ok] / H[ok]
    return np.real(np.fft.ifft2(F_hat))


def wiener_filter(g, H, nsr):
    """Wiener filter: F_hat = conj(H) / (|H|^2 + Sn/Sf) * G.

    nsr = Sn/Sf is the noise-to-signal power ratio. It can be one number
    (constant K) or an array with one value per frequency.
    """
    G = np.fft.fft2(g)
    W = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return np.real(np.fft.ifft2(W * G))


def true_nsr(f, sigma):
    """Sn/Sf computed with the ORIGINAL image (only possible in a test).

    Sf = |F|^2 / (M N)   (periodogram of the original image)
    Sn = sigma^2         (white noise has a flat spectrum)
    """
    Sf = np.abs(np.fft.fft2(f)) ** 2 / f.size
    Sf = np.maximum(Sf, 1e-10)                 # avoid division by zero
    return sigma ** 2 / Sf


# ------------------------------------------------------------------
# Estimating sigma and Sf from the degraded image only
# ------------------------------------------------------------------

def estimate_sigma(g):
    """Estimate the noise standard deviation from the image itself.

    d = (a - b - c + e) / 2 for every 2x2 block [[a, b], [c, e]].
    Smooth image parts cancel in d, so d is mostly noise, with variance
    sigma^2. We use the median instead of the mean because the median is
    not disturbed by the few large values at edges in the image.
    For Gaussian noise: median(|d|) = 0.6745 * sigma.
    """
    d = (g[:-1, :-1] - g[:-1, 1:] - g[1:, :-1] + g[1:, 1:]) / 2
    return np.median(np.abs(d)) / 0.6745


def frequency_radius(shape):
    """Distance of every DFT bin to frequency 0, in cycles per pixel."""
    u = np.fft.fftfreq(shape[0])
    v = np.fft.fftfreq(shape[1])
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def periodogram(g):
    """Periodogram |G|^2 / (M N) of the image (mean removed)."""
    return np.abs(np.fft.fft2(g - g.mean())) ** 2 / g.size


def smooth_periodogram(P, width=5):
    """Average every periodogram value with its neighbours (width x width).

    The raw periodogram is very noisy (its standard deviation is as large
    as its mean). Averaging 25 neighbours lowers the variance about 25
    times, at the cost of frequency resolution.
    """
    total = np.zeros_like(P)
    half = width // 2
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            total += np.roll(P, (di, dj), axis=(0, 1))
    return total / width ** 2


def estimate_Sf(g, H, sigma):
    """Estimate the image spectrum Sf from the degraded image.

    Model for natural images: Sf(rho) = A / rho^alpha (a power law).
    Model for the degraded image: Sg = |H|^2 Sf + sigma^2,
    so  Sf = (Sg - sigma^2) / |H|^2.
    We compute that only at frequencies where it is reliable (|H| not too
    small and Sg clearly above the noise level), then fit a straight line
    log(Sf) = log(A) - alpha * log(rho).
    """
    rho = frequency_radius(g.shape)
    Sg = smooth_periodogram(periodogram(g))
    H2 = np.abs(H) ** 2

    good = (rho > 0) & (H2 > 0.1) & (Sg > 2 * sigma ** 2)
    Sf_points = (Sg[good] - sigma ** 2) / H2[good]
    slope, intercept = np.polyfit(np.log(rho[good]), np.log(Sf_points), 1)
    alpha, A = -slope, np.exp(intercept)

    rho_safe = np.maximum(rho, rho[good].min())  # avoid 1/0 at rho = 0
    return A / rho_safe ** alpha, alpha


# ------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------

def mse(f, f_hat):
    return np.mean((f - f_hat) ** 2)


def psnr(f, f_hat):
    return 10 * np.log10(255 ** 2 / mse(f, f_hat))


# ------------------------------------------------------------------
# Realistic (non-circular) blur and edge tapering
# ------------------------------------------------------------------

def blur_realistic(f, length, angle_deg, pad=30):
    """Blur without wrap-around: extend the image by repeating its border
    pixels, blur, and cut the original size back out."""
    f_big = np.pad(f, pad, mode="edge")
    H_big = np.fft.fft2(motion_psf(length, angle_deg, f_big.shape))
    return blur(f_big, H_big)[pad:-pad, pad:-pad]


def edge_window(n, width):
    """1-D window: 1 in the middle, smooth cosine ramp from 0 to 1 over
    `width` samples at both ends."""
    w = np.ones(n)
    for i in range(width):
        value = 0.5 - 0.5 * np.cos(np.pi * (i + 0.5) / width)
        w[i] = value
        w[n - 1 - i] = value
    return w


def taper_edges(g, width):
    """Fade the image smoothly to its mean value near the borders, so the
    left/right and top/bottom edges match when the DFT wraps around."""
    w = np.outer(edge_window(g.shape[0], width), edge_window(g.shape[1], width))
    return w * g + (1 - w) * g.mean()

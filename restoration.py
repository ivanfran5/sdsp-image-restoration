"""Image degradation and restoration: motion blur, Gaussian noise,
inverse filter and Wiener filter.

Data model:  g = h * f + n   <=>   G = H F + N   (circular convolution, DFT)
"""
import numpy as np
import scipy.io


def load_images(path):
    """Load I1 and I2 from the .mat file as float arrays (0-255)."""
    data = scipy.io.loadmat(path)
    return {"I1": data["I1"].astype(float), "I2": data["I2"].astype(float)}


# ------------------------------------------------------------------
# Degradation
# ------------------------------------------------------------------

def motion_psf(length, angle_deg, shape):
    """Line-shaped motion-blur kernel, centred at pixel (0, 0) of an
    image-sized array (negative indices wrap around)."""
    psf = np.zeros(shape)
    angle = np.deg2rad(angle_deg)
    for k in range(length):
        t = k - (length - 1) / 2
        row = int(round(-t * np.sin(angle)))
        col = int(round(t * np.cos(angle)))
        psf[row, col] += 1
    return psf / psf.sum()


def blur(f, H):
    """Circular blur: multiply the spectrum of f by H."""
    return np.real(np.fft.ifft2(np.fft.fft2(f) * H))


def add_noise(f, sigma, rng):
    """Add white Gaussian noise with standard deviation sigma."""
    return f + sigma * rng.standard_normal(f.shape)


def blur_realistic(f, length, angle_deg, pad=30):
    """Blur without wrap-around: pad with border pixels, blur, crop."""
    f_big = np.pad(f, pad, mode="edge")
    H_big = np.fft.fft2(motion_psf(length, angle_deg, f_big.shape))
    return blur(f_big, H_big)[pad:-pad, pad:-pad]


# ------------------------------------------------------------------
# Restoration filters
# ------------------------------------------------------------------

def inverse_filter(g, H, eps):
    """F_hat = G / H, set to zero where |H| <= eps."""
    G = np.fft.fft2(g)
    F_hat = np.zeros_like(G)
    ok = np.abs(H) > eps
    F_hat[ok] = G[ok] / H[ok]
    return np.real(np.fft.ifft2(F_hat))


def wiener_filter(g, H, nsr):
    """F_hat = conj(H) / (|H|^2 + Sn/Sf) * G; nsr is a scalar K or an array."""
    G = np.fft.fft2(g)
    W = np.conj(H) / (np.abs(H) ** 2 + nsr)
    return np.real(np.fft.ifft2(W * G))


def true_nsr(f, sigma):
    """Sn/Sf using the periodogram of the original image (reference only)."""
    Sf = np.abs(np.fft.fft2(f)) ** 2 / f.size
    Sf = np.maximum(Sf, 1e-10)
    return sigma ** 2 / Sf


# ------------------------------------------------------------------
# Estimating sigma and Sf from the degraded image
# ------------------------------------------------------------------

def estimate_sigma(g):
    """Noise std from the median of the diagonal 2x2 pixel difference."""
    d = (g[:-1, :-1] - g[:-1, 1:] - g[1:, :-1] + g[1:, 1:]) / 2
    return np.median(np.abs(d)) / 0.6745


def frequency_radius(shape):
    """Distance of every DFT bin to frequency 0 (cycles/pixel)."""
    u = np.fft.fftfreq(shape[0])
    v = np.fft.fftfreq(shape[1])
    U, V = np.meshgrid(u, v, indexing="ij")
    return np.sqrt(U ** 2 + V ** 2)


def periodogram(g):
    """|G|^2 / (M N) of the mean-removed image."""
    return np.abs(np.fft.fft2(g - g.mean())) ** 2 / g.size


def smooth_periodogram(P, width=5):
    """Average each value with its width x width neighbours (lower variance)."""
    total = np.zeros_like(P)
    half = width // 2
    for di in range(-half, half + 1):
        for dj in range(-half, half + 1):
            total += np.roll(P, (di, dj), axis=(0, 1))
    return total / width ** 2


def estimate_Sf(g, H, sigma):
    """Fit Sf = A / rho^alpha to (Sg - sigma^2) / |H|^2 on a log-log scale."""
    rho = frequency_radius(g.shape)
    Sg = smooth_periodogram(periodogram(g))
    H2 = np.abs(H) ** 2

    good = (rho > 0) & (H2 > 0.1) & (Sg > 2 * sigma ** 2)
    Sf_points = (Sg[good] - sigma ** 2) / H2[good]
    slope, intercept = np.polyfit(np.log(rho[good]), np.log(Sf_points), 1)
    alpha, A = -slope, np.exp(intercept)

    rho_safe = np.maximum(rho, rho[good].min())
    return A / rho_safe ** alpha, alpha


# ------------------------------------------------------------------
# Edge tapering (windowing)
# ------------------------------------------------------------------

def edge_window(n, width):
    """1 in the middle, cosine ramp 0 -> 1 over `width` samples at both ends."""
    w = np.ones(n)
    for i in range(width):
        value = 0.5 - 0.5 * np.cos(np.pi * (i + 0.5) / width)
        w[i] = value
        w[n - 1 - i] = value
    return w


def taper_edges(g, width):
    """Fade the image to its mean value near the borders."""
    w = np.outer(edge_window(g.shape[0], width), edge_window(g.shape[1], width))
    return w * g + (1 - w) * g.mean()


# ------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------

def mse(f, f_hat):
    return np.mean((f - f_hat) ** 2)


def psnr(f, f_hat):
    return 10 * np.log10(255 ** 2 / mse(f, f_hat))

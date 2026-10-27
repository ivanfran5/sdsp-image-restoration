# SDSP image restoration (inverse and Wiener filtering)

Course project: restore grayscale images degraded by motion blur and Gaussian noise.

- `img_restoration.mat`: the two test images `I1` (220x220) and `I2` (256x256), uint8, 0-255.
- `restoration.py`: degradation model, inverse filter (thresholded), Wiener filter, estimation of the noise level and image spectrum from the degraded image, edge tapering, MSE/PSNR. Only `numpy.fft.fft2/ifft2` are used; the filters are implemented by hand.
- `image_restoration.ipynb`: data model and all experiments, with results and observations:
  1. motion-blur kernel and its transfer function
  2. the three distortions (blur, noise, blur + noise)
  3. inverse filter and the threshold eps
  4. Wiener filter with a constant K
  5. Wiener filter with the true image spectrum
  6. Wiener filter with sigma and the spectrum estimated from the degraded image
  7. results table for all distortions and both images
  8. PSNR versus noise level
  9. different blur lengths and angles
  10. restoring with a wrongly assumed blur kernel
  11. realistic blur at the borders: edge tapering and padding
  12. summary: which filter works best when, and the problems we solved

The notebook uses a fixed random seed, so every run gives the same numbers (about 1 minute).

```
pip install -r requirements.txt
jupyter notebook image_restoration.ipynb
```

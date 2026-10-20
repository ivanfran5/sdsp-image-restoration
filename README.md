# SDSP image restoration (inverse and Wiener filtering)

Course project: restore grayscale images degraded by motion blur and Gaussian noise.

- `img_restoration.mat`: the two test images `I1` (220x220) and `I2` (256x256), uint8, 0-255.
- `restoration.py`: degradation model, inverse filter (thresholded), Wiener filter, MSE/PSNR. Only `numpy.fft.fft2/ifft2` are used; the filters are implemented by hand.
- `image_restoration.ipynb`: experiments (blur only, noise only, blur + noise, both images).

```
pip install -r requirements.txt
jupyter notebook image_restoration.ipynb
```

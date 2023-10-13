# My Research Activities of Underwater Image Processing (UIP)

## Underwater Image Enhancement (UIE)

### [UIE models](./models/img_enhance_model.py)

#### 1.ImgEnhanceModel

Loss Functions:
- MAE(L1) Loss
- PSNR Loss
- SSIM Loss
    ```math
    \mathcal{L}_{SSIM} = \frac{1 - SSIM\left(\mathbf{\hat{Y}}, \mathbf{Y}\right)}{2}
    ```

#### 2.ImgEnhanceModel2

Loss Functions:
- L1 Charbonnier Loss
- PSNR Loss
- SSIM Loss

#### 3.ImgEnhanceModel3

Loss Functions:
- MAE(L1) Loss
- PSNR Loss
- SSIM Loss
- Fourier Domain Loss
    
    $$ \mathcal{L}_{Four} = {\ell}_{1}(amp(\mathbf{\hat{Y}}), amp(\mathbf{Y})) + {\ell}_{1}(pha(\mathbf{\hat{Y}}), pha(\mathbf{Y})) $$

#### 4.ImgEnhanceModel4

Loss Functions:
- MAE(L1) Loss
- PSNR Loss
- SSIM Loss
- Fourier Domain Loss
    $$ \mathcal{L}_{Four} = {\ell}_{1}(amp(\mathbf{\hat{Y}}), amp(\mathbf{Y})) + {\ell}_{1}(pha(\mathbf{\hat{Y}}), pha(\mathbf{Y})) $$
- Edge Detection Loss
    $$ \mathcal{L}_{Edge} = {\ell}_{1}(Canny(\mathbf{\hat{Y}}), Canny(\mathbf{Y})) $$

#### 5.ImgEnhanceModel5

Loss Functions:
- MAE(L1) Loss
- PSNR Loss
- SSIM Loss
- Fourier Domain Loss
    $$ \mathcal{L}_{Four} = {\ell}_{1}(fft(\mathbf{\hat{Y}}), fft(\mathbf{Y})) $$
- Edge Detection Loss
    $$ \mathcal{L}_{Edge} = {\ell}_{1}(Canny(\mathbf{\hat{Y}}), Canny(\mathbf{Y})) $$

### Networks

#### 1.RANet
#### 2.RANet2
#### 3.RANet3
#### 4.RANet4
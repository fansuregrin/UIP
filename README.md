# Underwater Image Processing (UIP)


## Models
<table>
<tr>
  <td>ImgEnhanceModel</td>
  <td>ImgEnhanceModel2</td>
  <td>ImgEnhanceModel3</td>
  <td>ImgEnhanceModel4</td>
  <td>ImgEnhanceModel5</td>
</tr>
<tr>
  <td>ImgEnhanceModel6</td>
  <td>ImgEnhanceModel7</td>
  <td>ImgEnhanceModel8</td>
  <td>ImgEnhanceModel9</td>
  <td>AquaticMamba</td>
</tr>
</table>


## Networks
<table>
<tr>
    <td><a href="./networks/unet/">UNet</a></td>
    <td><a href="./networks/fcn/">FCN</a></td>
    <td><a href="./networks/ege_unet/">EGE_UNet</a></td>
    <td><a href="./networks/ugan/">UGAN</a></td>
    <td><a href="./networks/waternet/">WaterNet</a></td>
</tr>
<tr>
    <td><a href="./networks/ra_net/">RA</a></td>
    <td><a href="./networks/erd/">ERD</a></td>
    <td><a href="./networks/vg_unet/">VGUNet</a></td>
    <td><a href="./networks/mimo_swinT_unet/">MimoSwinTUNet</a></td>
    <td><a href="./networks/aquatic_mamba/">Aquatic Mamba</a></td>
</tr>
</table>

## Citations
If you use this repository, please cite the following papers:
```bibtex
@InProceedings{raune-net,
    author="Peng, Wangzhen
        and Zhou, Chenghao
        and Hu, Runze
        and Cao, Jingchao
        and Liu, Yutao",
    editor="Zhai, Guangtao
        and Zhou, Jun
        and Ye, Long
        and Yang, Hua
        and An, Ping
        and Yang, Xiaokang",
    title="RAUNE-Net: A Residual and Attention-Driven Underwater Image Enhancement Method",
    booktitle="Digital Multimedia Communications",
    year="2024",
    publisher="Springer Nature Singapore",
    address="Singapore",
    pages="15--27",
    isbn="978-981-97-3623-2"
}

@article{erd,
    author={Cao, Jingchao and Peng, Wangzhen and Liu, Yutao and Dong, Junyu and Callet, Patrick Le and Kwong, Sam},
    journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
    title={ERD: Encoder-Residual-Decoder Neural Network for Underwater Image Enhancement}, 
    year={2025},
    volume={},
    number={},
    pages={1-1},
    keywords={Image color analysis;Feature extraction;Transformers;Image quality;Training;Imaging;Image restoration;Image enhancement;Image edge detection;Degradation;Underwater image enhancement;Deep neural network;Residual learning;Attention;Fourier transform},
    doi={10.1109/TCSVT.2025.3556203}
}
```
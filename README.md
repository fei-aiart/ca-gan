# CA-GAN

[[Project Page\]](https://github.com/fei-hdu/ca-gan/) [[Paper\]](https://arxiv.org/abs/1712.00899) 

Pytorch implementation for composition-aided face sketch-photo synthesis.

> **Composition-Aided Face Photo-Sketch Synthesis.**
>
> Fei Gao, Shengjie Shi, Jun Yu,  Dacheng Tao, and Qingming Huang

### Results

We list a number of results of composition-aided GAN (CA-GAN) and stacked CA-GAN (SCA-GAN) on the CUHK, CUFSF, and VIPSL-FS databases.  

- `Photo2Sketch`: Synthesize a sketch image from a photo.
- `Sketch2Photo`: Synthesize a photo image from a sketch.
- `cross dataset`: Models are trained on CUHK dataset and tested on VIPSL dataset.

------

- `(WANG)`: WANG's dividing method. Other images are synthesized with the  model that are trained on dataset of our dividing method. 

------

- Each image has five faces (sketches or photos). 
  - 1: Input image
  - 2: Ground truth
  - 3: Synthesized sketch/photo by cGAN
  - 4: Synthesized sketch/photo by CA-GAN
  - 5: Synthesized sketch/photo by SCA-GAN

- **The full results are available at [BaiduCloud][https://pan.baidu.com/s/1PnzNYdwl6Cd2V5gg00ehQQ]**, password: `rhd1`

  ​

###Example results

1) Synthesized sketches of Chinese celebrities. (a) Photo, (b) cGAN, (c) CA-GAN, (d) SCA-GAN.

[![img](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_celeb_sketch.jpg)](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_celeb_sketch.jpg)

2) Examples of high-resolution synthesized face sketches on the VIPSL-FS database. (a) Photo, (b) cGAN, (c) CA-GAN, (d) SCA-GAN, and (e) Sketch drawn by artist.

[![img](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_sketch_vipsl.jpg)](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_sketch_vipsl.jpg)

3) Examples of synthesized high-resolution face photos on the VIPSL-FS database. (a) Sketch drawn by artist, (b) cGAN, (c) CA-GAN, (d) SCA-GAN, and (e) ground truth photo.

[![img](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_photo_vipsl.jpg)](https://github.com/fei-hdu/ca-gan/blob/master/Examples/fig_photo_vipsl.jpg)



### Codes



### Datasets

- CUHK
- CUFSF
- VIPSL-FS

### Citation

If you find this useful for your research, please use the following.

```
@article{gao2017ca-gan,
	title = {Composition-Aided Face Photo-Sketch Synthesis},
	author = {Fei Gao, Shengjie Shi, Jun Yu,  Dacheng Tao, and Qingming Huang},
	booktitle = {arXiv:1712.00899},
	year = {2017},
}
```

### Acknowledgements

This code borrows heavily from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.

The results of exisiting methods (except cGAN) are provided by Nannan Wang and Chunlei Peng. 

- Wang et al. at: http://www.ihitworld.com/RSLCR.html.
- Peng: http://chunleipeng.com/TNNLS2015_MrFSPS.html.
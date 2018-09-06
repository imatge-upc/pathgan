# PathGan: Visual Scan-path Prediction with Generative Adversarial Networks

| ![Marc Assens][MarcAssens-photo] | ![Kevin McGuinness][KevinMcGuinness-photo] | ![Xavier Giro-i-Nieto][XavierGiro-photo]| ![Noel O'Connor][NoelOConnor-photo] |
|:-:|:-:|:-:|:-:|
| [Marc Assens][MarcAssens-web]  | [Kevin McGuinness][KevinMcGuinness-web]  | [Xavier Giro-i-Nieto][XavierGiro-web] | [Noel O'Connor][NoelOConnor-web]   |

[MarcAssens-web]: https://www.linkedin.com/in/marc-assens-reina-5b1090bb/
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro

[MarcAssens-photo]: https://github.com/massens/saliency-360salient-2017/raw/master/authors/foto_carnet_dublin.jpg "Marc Assens"
[KevinMcGuinness-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-salgan-2017/junting/authors/Kevin160x160%202.jpg?token=AFOjyZmLlX3ZgpkNe60Vn3ruTsq01rD9ks5YdAaiwA%3D%3D "Kevin McGuinness"
[XavierGiro-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/XavierGiro.jpg "Xavier Giro-i-Nieto"
[NoelOConnor-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/NoelOConnor.jpg "Noel O'Connor"


A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-gpi] |
|:-:|:-:|:-:|
| [Insight Centre for Data Analytics][insight-web] | [Dublin City University (DCU)][dcu-web] | [UPC Image Processing Group][gpi-web] |

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"


## Abstract

We introduce PathGAN, a deep neural network for visual scanpath prediction trained on adversarial examples. A visual scanpath is defined as the sequence of fixation points over an image defined by a human observer with its gaze. PathGAN is composed of two parts, the generator and the discriminator. Both parts extract features from images using off-the-shelf networks, and train recurrent layers to generate or discriminate scanpaths accordingly. In scanpath prediction, the stochastic nature of the data makes it very difficult to generate realistic predictions using supervised learning strategies, but we adopt adversarial training as a suitable alternative. Our experiments prove how PathGAN improves the state of the art of visual scanpath prediction on the iSUN and Salient360! datasets. 

## Publication

Find the pre-print version of our work on [arXiv](https://arxiv.org/abs/1809.00567).

![Image of the paper](https://github.com/imatge-upc/pathgan/raw/master/figs/paper.png)

Please cite with the following Bibtex code:

```
@inproceedings{Assens2018pathgan,
title={PathGAN: Visual Scanpath Prediction with Generative Adversarial Networks},
author={Marc Assens, Xavier Giro-i-Nieto, Kevin McGuinness, Noel E. O'Connor},
journal={ECCV Workshop on Egocentric Perception, Interaction and Computing (EPIC)},
year={2018}
}
```

You may also want to refer to our publication with the more human-friendly Chicago style:

*Marc Assens, Xavier Giro-i-Nieto, Kevin McGuinness, Noel E. O’Connor. “PathGAN: Visual Scanpath Prediction with Generative Adversarial Networks”, ECCV Workshop on Egocentric Perception, Interaction and Computing (EPIC), 2018.*



## Models

The model is composed by two deep neural networks, the generator and the discriminator, whose combined efforts aim at predicting a realistic scanpath from a given image.

Model Architecture:
![architecture-fig](https://github.com/imatge-upc/pathgan/raw/master/figs/model.png)

* [[Scan-path generator model (100 MB)]]()




## Examples
We provide examples of predicted object sequences for two datasets.

##### 1. iSun

![ex-isun](https://github.com/imatge-upc/pathgan/raw/master/figs/ex_isun.png)

##### 2. Salient360!



![ex-360](https://github.com/imatge-upc/pathgan/raw/master/figs/ex_360.png)


The big dot indicates the first fixation of the scanpath. 


## Software frameworks: Keras

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [Theano](http://deeplearning.net/software/theano/).

## Acknowledgements
We especially want to thank our technical support team:

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"
[JosepPujal-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/JosepPujal.jpg "Josep Pujal"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno
[JosepPujal-web]: https://imatge.upc.edu/web/people/josep-pujal

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeForce GTX [Titan Z](http://www.nvidia.com/gtx-700-graphics-cards/gtx-titan-z/) and [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image Processing Group at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |
|  This work has been developed in the framework of projects TEC2013-43935-R and TEC2016-75976-R, financed by the Spanish Ministerio de Economía y Competitividad and the European Regional Development Fund (ERDF).  | ![logo-spain] | 

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"
[logo-spain]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/MEyC.png "Logo of Spanish government"

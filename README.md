# DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks

<p align="center">
  <a href="Give a link here"><img src="https://img.shields.io/badge/paper-report-red"/></a>
  <a href="Give a link here"><img src="https://img.shields.io/github/license/thudm/cogdl"/></a>
  <a href="Give a link here" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

<!--PUT HERE SOME QUALITATIVE RESULTS IN THE ASSETS FOLDER-->
<!--YOU CAN PUT ALSO IN THE GIF OR PNG FORMAT -->
<p float="center">
  <img src="assets/sample1.png" width="49%" />
  <img src="assets/sample2.png" width="49%" />
</p>

<!--PUT THE ANIMATED GIF VERSION OF THE DRUGGEN MODEL (Figure 1)-->
<p float="center">
  <img src="assets/druggen_figures(1).gif" width="98%" />
</p>

Check out our paper below for more details

> [**DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks
**](Give a link here),            
> [Atabey Ünlü](https://tr.linkedin.com/in/atabeyunlu), [Elif Çevrim](https://www.linkedin.com/in/elifcevrim/?locale=en_US), [Ahmet Sarıgün](https://asarigun.github.io/), [Heval Ataş](https://www.linkedin.com/in/heval-atas/), [Altay Koyaş](https://www.linkedin.com/in/altay-koya%C5%9F-8a6118a1/?originalSubdomain=tr), [Hayriye Çelikbilek](https://www.linkedin.com/in/hayriye-celikbilek/?originalSubdomain=tr), [Deniz Cansen Kahraman](https://www.linkedin.com/in/deniz-cansen-kahraman-6153894b/?originalSubdomain=tr), [Abdurrahman Olğaç](https://www.linkedin.com/in/aolgac/?originalSubdomain=tr), [Ahmet S. Rifaioğlu](https://saezlab.org/person/ahmet-sureyya-rifaioglu/), [Tunca Doğan](https://yunus.hacettepe.edu.tr/~tuncadogan/)     
> *Arxiv, 2020* 

## Features

<!--PUT HERE 1-2 SENTECE FOR METHOD WHICH SHOULD BE SHORT --> Pleaser refer to our [arXiv report](Give a link here) for further details.

This implementation:

- has the demo and training code for DrugGEN implemented in PyTorch Geometric,
- can design de novo drugs based on their protein interactions,
<!-- - supports both CPU and GPU inference (though GPU is way faster), -->
<!-- ADD HERE SOME FEATURES FOR DRUGGEN & SUMMARIES & BULLET POINTS -->


<!-- ADD THE ANIMATED GIF VERSION OF THE GAN1 AND GAN2 -->
| First Generator                                                                                                | Second Generator                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![FirstGAN](assets/generator_1.gif) | ![SecondGAN](assets/generator_2.gif) |

## Overview
We provide the implementation of the DrugGEN in PyTorch Geometric framework, along with scripts to generate and run. The repository is organised as follows:

- ```data``` contains:

## Datasets

<!-- ADD SOME INFO HERE -->

## Updates

- 00/00/2022: First version script of DrugGEN is released.

## Getting Started
DrugGEN has been implemented and tested on Ubuntu 18.04 with python >= 3.9. It supports both GPU and CPU inference.
If you don't have a suitable device, try running our Colab demo. 

Clone the repo:
```bash
git clone https://github.com/asarigun/DrugGEN.git
```

Install the requirements using `virtualenv` or `conda`:
```bash
# pip
source install/install_pip.sh

# conda
source install/install_conda.sh
```
## Running the Demo
You could try Google Colab if you don't already have a suitable environment for running this project.
It enables cost-free project execution in the cloud. You can use the provided notebook to try out our Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Give a link here)

## Training

<!--ADD HERE TRAINING COMMANDS WITH EXPLAINATIONS-->

## Citation
<!--ADD BIBTEX AFTER THE PUBLISHING-->

## License
<!--ADD LICENSE TERMS AND LICENSE FILE AND GIVE A LINK HERE-->

## References

In each file, we indicate whether a function or script is imported from another source. Here are some excellent sources from which we benefit: 
<!--ADD THE REFERENCES THAT WE USED DURING THE IMPLEMENTATION-->
- First GAN is borrowed from [MolGAN](https://github.com/yongqyu/MolGAN-pytorch)
- 
-

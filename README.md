# DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks



<p align="center">
  <a href="Give a link here"><img src="https://img.shields.io/badge/paper-report-red"/></a>
  <a href="Give a link here"><img src="https://img.shields.io/github/license/thudm/cogdl"/></a>
  <a href="Give a link here" alt="license"><img src="https://colab.research.google.com/assets/colab-badge.svg"/></a>
</p>

<!--PUT HERE SOME QUALITATIVE RESULTS IN THE ASSETS FOLDER-->
<!--YOU CAN PUT ALSO IN THE GIF OR PNG FORMAT -->
<!--<p float="center">
  <img src="assets/sample1.png" width="49%" />
  <img src="assets/sample2.png" width="49%" />
</p>-->

## Abstract

Discovering novel drug candidate molecules is one of the most fundamental and critical steps in drug development. Generative deep learning models, which create synthetic data given a probability distribution, have been developed with the purpose of picking completely new samples from a partially known space. Generative models offer high potential for designing de novo molecules; however, in order for them to be useful in real-life drug development pipelines, these models should be able to design target-specific molecules, which is the next step in this field. In this study, we propose a novel generative system, DrugGEN, for de novo design of drug candidate molecules that interact with selected target proteins. The proposed system represents compounds and protein structures as graphs and processes them via serially connected two generative adversarial networks comprising graph transformers. DrugGEN is implemented with five independent models, each with a unique sample generation routine. The system is trained using a large dataset of compounds from ChEMBL and target-specific bioactive molecules, to design effective and specific inhibitory molecules against the AKT1 protein, effective targeting of which has critical importance for developing treatments against various types of cancer. DrugGEN has a competitive or better performance against other methods on fundamental benchmarks. To assess the target-specific generation performance, we conducted further in silico analysis with molecular docking and deep learning-based bioactivity prediction. Their results indicate that de novo molecules have high potential for interacting with the AKT1 protein structure in the level of its native ligand. DrugGEN can be used to design completely novel and effective target-specific drug candidate molecules for any druggable protein, given the target features and a dataset of experimental bioactivities.

<!--Check out our paper below for more details

> [**DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks
**](link here),            
> [Atabey Ünlü](https://tr.linkedin.com/in/atabeyunlu), [Elif Çevrim](https://www.linkedin.com/in/elifcevrim/?locale=en_US), [Ahmet Sarıgün](https://asarigun.github.io/), [Heval Ataş](https://www.linkedin.com/in/heval-atas/), [Altay Koyaş](https://www.linkedin.com/in/altay-koya%C5%9F-8a6118a1/?originalSubdomain=tr), [Hayriye Çelikbilek](https://www.linkedin.com/in/hayriye-celikbilek/?originalSubdomain=tr), [Deniz Cansen Kahraman](https://www.linkedin.com/in/deniz-cansen-kahraman-6153894b/?originalSubdomain=tr), [Abdurrahman Olğaç](https://www.linkedin.com/in/aolgac/?originalSubdomain=tr), [Ahmet S. Rifaioğlu](https://saezlab.org/person/ahmet-sureyya-rifaioglu/), [Tunca Doğan](https://yunus.hacettepe.edu.tr/~tuncadogan/)     
> *Arxiv, 2020* -->

<!--PUT THE ANIMATED GIF VERSION OF THE DRUGGEN MODEL (Figure 1)-->
<p float="center">
  <img src="assets/DrugGEN_Figure1.gif" width="63%" />
</p>



## Features

<!--PUT HERE 1-2 SENTECE FOR METHOD WHICH SHOULD BE SHORT Pleaser refer to our [arXiv report](link here) for further details.--> 

This implementation:

- has the training code for DrugGEN implemented in PyTorch,
<!-- - supports both CPU and GPU inference (though GPU is way faster), -->
<!-- ADD HERE SOME FEATURES FOR DRUGGEN & SUMMARIES & BULLET POINTS -->


<!-- ADD THE ANIMATED GIF VERSION OF THE GAN1 AND GAN2 -->
| First Generator                                                                                                | Second Generator                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![FirstGAN](assets/generator_1_mod.gif) | ![SecondGAN](assets/generator_2_mod.gif) |


## Preliminary results (generated molecules)

| ChEMBL-25                                                                                                | ChEMBL-45                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![ChEMBL_25](assets/molecule_1.png) | ![ChEMBL_45](assets/molecule_2.png) |

## Overview
We provide the implementation of the DrugGEN, along with scripts from PyTorch Geometric framework to generate and run. The repository is organised as follows:

```data``` contains: 
- **Raw dataset files**, which should be text files containing SMILES strings only. Raw datasets preferably should not contain stereoisomeric SMILES to prevent Hydrogen atoms to be included in the final graph data. 
- Constructed **graph datasets** (.pt) will be saved in this folder along with atom and bond encoder/decoder files (.pk).

```experiments``` contains: 
- ```logs``` folder. Model loss and performance metrics will be saved in this directory in seperate files for each model. 
- ```results``` folder. Tensorboard files will be saved here if TensorBoard is used.
- ```models``` folder. Models will be saved in this directory at last or preferred steps. 
- ```samples``` folder. Molecule samples will be saved in this folder.

**Python scripts are:**

- ```layers.py``` file contains **Transformer Encoder**, **Transformer Decoder**, **PNA** (Corso et al., 2020), and **Graph Convolution Network** implementations.  
- ```main.py``` contains arguments and this file is used to run the model.   
- ```models.py``` has the implementation of the **Generators** and **Discriminators** which are used in GAN1 and GAN2.  
- ```new_dataloader.py``` constructs the graph dataset from given raw data. Uses PyG based data classes.  
- ```trainer.py``` is the training and testing file for the model. Workflow is constructed in this file.   
- ```utils.py``` contains performance metrics from several other papers and some unique implementations. (De Cao et al, 2018; Polykovskiy et al., 2020)  

## Datasets
Three different data types (i.e., compound, protein, and bioactivity) were retrieved from various data sources to train our deep generative models. GAN1 module requires only compound data while GAN2 requires all of three data types including compound, protein, and bioactivity.
- **Compound data** includes atomic, physicochemical, and structural properties of real drug and drug candidate molecules. Small-scale QM9 compound dataset was used for the first version of GAN1 module for easy and fast processing. It consists of 133,885 stable organic molecules with a maximum of 9 atoms and containing only C, O, N, F heavy atoms. Then, GAN1 was retrained on large-scale ChEMBL compound dataset to improve model performance. Two versions of ChEMBL dataset were created considering heavy atom distribution of the dataset. Each is limited to a maximum atomic number of 25 and 45, and referred as ChEMBL-m25 and ChEMBL-m45, respectively. 12 heavy atom types were selected for both datasets considering the atom frequency, which are C, O, N, F, Ca, K, Br, B, S, P, Cl, and As. The number of molecules in ChEMBL-m25 and ChEMBL-m45 datasets is  651,509 and 1,588,865, respectively.
- **Protein data** was retrieved from Protein Data Bank (PDB) in biological assembly format, and the coordinates of protein-ligand complexes were used to construct the binding sites of proteins from the bioassembly data. The atoms of protein residues within a maximum distance of 9 A from all ligand atoms were recorded as binding sites. GAN2 was trained for generating compounds specific to the target protein AKT1, which is a member of serine/threonine-protein kinases and involved in many cancer-associated cellular processes including metabolism, proliferation, cell survival, growth and angiogenesis. Binding site of human AKT1 protein was generated from the kinase domain (PDB: 4GV1). 
- **Bioactivity data** of AKT target protein was retrieved from large-scale ChEMBL bioactivity database. It contains ligand interactions of human AKT1 (CHEMBL4282) protein with a pChEMBL value equal to or greater than 6 (IC50 <= 1 µM) as well as SMILES information of these ligands. The dataset was extended by including drug molecules from DrugBank database known to interact with human AKT proteins. Thus, a total of 3,251 bioactivity data points were obtained for training the AKT-specific generative model. To enhance the size of the bioactivity dataset, we also obtained two alternative versions by incorporating ligand interactions of protein members in non-specific serine/threonine kinase (STK) and kinase families.

More details on the construction of datasets can be found in our paper referenced above.

<!-- ADD SOME INFO HERE -->

## Updates

- 00/00/2022: First version script of DrugGEN is released.

## Getting Started
DrugGEN has been implemented and tested on Ubuntu 18.04 with python >= 3.9. It supports both GPU and CPU inference.
If you don't have a suitable device, try running our Colab demo. 

Clone the repo:
```bash
git clone https://github.com/HUBioDataLab/DrugGEN.git
```
** Please check the requirements.txt file to see dependencies. **

## Running the Demo
You could try Google Colab if you don't already have a suitable environment for running this project.
It enables cost-free project execution in the cloud. You can use the provided notebook to try out our Colab demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](Give a link here)

## Training

```bash
conda activate druggen

# Download the raw files

bash dataset_download.sh

# DrugGEN can be trained with a one-liner

python DrugGEN/main.py --submodel --mode="train" --device="cuda" --raw_file="DrugGEN/data/chembl_smiles.smi" --dataset_file="chembl45.pt" -- drug_raw_file="drug_smies.smi" --drug_dataset_file="drugs.pt" --max_atom=45
```

** Please find the arguments in the **main.py** file. Explanation of the commands can be found below.

```bash
Model arguments:
  --submodel SUBMODEL       Choose the submodel for training
  --act ACT                 Activation function for the model
  --z_dim Z_DIM             Prior noise for the first GAN
  --max_atom MAX ATOM       Maximum atom number for molecules must be specified
  --lambda_gp LAMBDA_GP     Gradient penalty lambda multiplier for the first GAN
  --dim DIM                 Dimension of the Transformer models for both GANs
  --depth DEPTH             Depth of the Transformer model from the first GAN
  --heads HEADS             Number of heads for the MultiHeadAttention module from the first GAN
  --dec_depth DEC_DEPTH     Depth of the Transformer model from the second GAN
  --dec_heads DEC_HEADS     Number of heads for the MultiHeadAttention module from the second GAN
  --mlp_ratio MLP_RATIO     MLP ratio for the Transformers
  --dis_select DIS_SELECT   Select the discriminator for the first and second GAN
  --init_type INIT_TYPE     Initialization type for the model
  --dropout DROPOUT         Dropout rate for the model
Training arguments:
  --batch_size BATCH_SIZE   Batch size for the training
  --epoch EPOCH             Epoch number for Training
  --warm_up_steps           Warm up steps for the first GAN
  --g_lr G_LR               Learning rate for G
  --g2_lr G2_LR             Learning rate for G2
  --d_lr D_LR               Learning rate for D
  --d2_lr D2_LR             Learning rate for D2      
  --n_critic N_CRITIC       Number of D updates per each G update
  --beta1 BETA1             Beta1 for Adam optimizer
  --beta2 BETA2             Beta2 for Adam optimizer 
  --clipping_value          Clipping value for the gradient clipping process
  --resume_iters            Resume training from this step for fine tuning if desired
Dataset arguments:      
  --features FEATURES       Additional node features (Boolean) (Please check new_dataloader.py Line 102)
```

<!--ADD HERE TRAINING COMMANDS WITH EXPLAINATIONS-->

## Citation
<!--ADD BIBTEX AFTER THE PUBLISHING-->

## License
Copyright (C) 2023 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

## References

In each file, we indicate whether a function or script is imported from another source. Here are some excellent sources from which we benefit: 
<!--ADD THE REFERENCES THAT WE USED DURING THE IMPLEMENTATION-->
- First GAN is inspired from [MolGAN](https://github.com/yongqyu/MolGAN-pytorch).
- [MOSES](https://github.com/molecularsets/moses) was used for performance calculation.
- [PyG](https://github.com/pyg-team/pytorch_geometric) was used to construct the custom dataset.

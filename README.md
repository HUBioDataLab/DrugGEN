# DrugGEN: Target Specific De Novo Design of Drug Candidate Molecules with Graph Transformer-based Generative Adversarial Networks



<p align="center">
  <a href="https://arxiv.org/abs/2302.07868"><img src="https://img.shields.io/badge/Pre--print-%40arXiv-ff0000"/></a>
  <a href="https://huggingface.co/spaces/HUBioDataLab/DrugGEN"><img src="https://img.shields.io/badge/model-HuggingFace-yellow?labelColor=gray&color=yellow"/></a>
  <a href="http://www.gnu.org/licenses/"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg"/></a>
  
</p>


## Updated Pre-print!

**Please see our most up-to-date document (pre-print) from 26.07.2024 here:**  [arXiv link](https://arxiv.org/abs/2302.07868)

&nbsp;
&nbsp;

## Abstract

Discovering novel drug candidate molecules is one of the most fundamental and critical steps in drug development. Generative deep learning models, which create synthetic data given a probability distribution, offer a high potential for designing de novo molecules. However, for them to be useful in real-life drug development pipelines, these models should be able to design drug-like and target-centric molecules. In this study, we propose an end-to-end generative system, DrugGEN, for the de novo design of drug candidate molecules that interact with intended target proteins. The proposed method represents molecules as graphs and processes them via a generative adversarial network comprising graph transformer layers. The system is trained using a large dataset of drug-like compounds and target-specific bioactive molecules to design effective inhibitory molecules against the AKT1 protein, which has critical importance for developing treatments against various types of cancer. We conducted further analysis with molecular docking and dynamics to assess the target-centric generation performance of the model, and attention score visualisation for interpretability. Results indicate that our de novo molecules have a high potential for interacting with the AKT1 protein at the level of its native ligands. Using the open-access DrugGEN codebase, it is possible to easily train models for other druggable proteins, given a dataset of experimentally known bioactive molecules.

Our up-to-date pre-print is shared [here](https://github.com/HUBioDataLab/DrugGEN/files/10828402/2302.07868.pdf)

<!--Check out our paper below for more details

> [**DrugGEN: Target Centric De Novo Design of Drug Candidate Molecules with Graph Generative Deep Adversarial Networks
**](link here),            
> [Atabey Ünlü](https://tr.linkedin.com/in/atabeyunlu), [Elif Çevrim](https://www.linkedin.com/in/elifcevrim/?locale=en_US), [Ahmet Sarıgün](https://asarigun.github.io/), [Heval Ataş](https://www.linkedin.com/in/heval-atas/), [Altay Koyaş](https://www.linkedin.com/in/altay-koya%C5%9F-8a6118a1/?originalSubdomain=tr), [Hayriye Çelikbilek](https://www.linkedin.com/in/hayriye-celikbilek/?originalSubdomain=tr), [Deniz Cansen Kahraman](https://www.linkedin.com/in/deniz-cansen-kahraman-6153894b/?originalSubdomain=tr), [Abdurrahman Olğaç](https://www.linkedin.com/in/aolgac/?originalSubdomain=tr), [Ahmet S. Rifaioğlu](https://saezlab.org/person/ahmet-sureyya-rifaioglu/), [Tunca Doğan](https://yunus.hacettepe.edu.tr/~tuncadogan/)     
> *Arxiv, 2020* -->

&nbsp;
&nbsp;

<!--PUT THE ANIMATED GIF VERSION OF THE DRUGGEN MODEL (Figure 1)-->
<p float="center">
  <img src="assets/DrugGEN_Figure.png" width="100%" />
</p>

**Fig. 1.** The schematic representation of the architecture of the DrugGEN model with powerful graph transformer encoder modules in both generator and discriminator networks. The generator module transforms the given input into a new molecular representation. The discriminator compares the generated de novo molecules to the known inhibitors of the given target protein, scoring them for their assignment to the classes of "real" and "fake" molecules (abbreviations; MLP: multi-layered perceptron, Norm: normalisation, Concat: concatenation, MatMul: matrix multiplication, ElementMul: element-wise multiplication, Mol. adj: molecule adjacency tensor, Mol. Anno: molecule annotation matrix, Upd: updated).

&nbsp;
&nbsp;

## Transformer Module

Given a random molecule *z*, **the generator** *G* (below) creates annotation and adjacency matrices of a supposed molecule. *G* processes the input by passing it through a multi-layer perceptron (MLP). The input is then fed to the graph transformer encoder module [Vaswani et al., (2017)](https://arxiv.org/abs/1706.03762), which has a depth of 1 encoder layers with 8 multi-head attention heads for each. In the graph transformer setting, *Q*, *K* and *V* are the variables representing the annotation matrix of the molecule. After the final products are created in the attention mechanism, both the annotation and adjacency matrices are forwarded to layer normalization and then summed with the initial matrices to create a residual connection. These matrices are fed to separate feedforward layers, and finally, given to the discriminator network *D* together with real molecules.


<!--PUT HERE 1-2 SENTECE FOR METHOD WHICH SHOULD BE SHORT Pleaser refer to our [arXiv report](link here) for further details.--> 


<!-- - supports both CPU and GPU inference (though GPU is way faster), -->
<!-- ADD HERE SOME FEATURES FOR DRUGGEN & SUMMARIES & BULLET POINTS -->


<!-- ADD THE ANIMATED GIF VERSION OF THE GAN1 AND GAN2 -->

<!-- |------------------------------------------------------------------------------------------------------------| -->
<!-- | ![FirstGAN](assets/DrugGEN_generator.gif) | -->

 <img src="/assets/transformer_module.gif" width="60%" height="60%"/>  

&nbsp;
&nbsp;

## Model Variations

- **DrugGEN** is the default model. The input of the generator is the real molecules (ChEMBL) dataset (to ease the learning process) and the discriminator compares the generated molecules with the real inhibitors of the given target protein.
- **DrugGEN-NoTarget** is the non-target-specific version of DrugGEN. This model only focuses on learning the chemical properties from the ChEMBL training dataset.

&nbsp;
&nbsp;

## Files & Folders

The DrugGEN repository is organized as follows:

### `data/`
- Contains raw dataset files and processed graph data for model training
- `encoders/` - Contains encoder files for molecule representation
- `decoders/` - Contains decoder files for molecule representation
- Format of raw dataset files should be text files containing SMILES strings only

### `src/`
Core implementation of the DrugGEN framework:
- `data/` - Data processing utilities
  - `dataset.py` - Handles dataset creation and loading
  - `utils.py` - Data processing helper functions
- `model/` - Model architecture components
  - `models.py` - Implementation of Generator and Discriminator networks
  - `layers.py` - Contains transformer encoder implementation
  - `loss.py` - Loss functions for model training
- `util/` - Utility functions
  - `utils.py` - Performance metrics and helper functions
  - `smiles_cor.py` - SMILES processing utilities

### `assets/`
- Graphics and figures used in documentation
- Contains model architecture diagrams and visualization resources
- Includes images of generated molecules and model animations

### `results/`
- Contains evaluation results and generated molecules
- `generated_molecules/` - Storage for molecules produced by the model
- `docking/` - Results from molecular docking analyses
- `evaluate.py` - Script for evaluating model performance

### `experiments/`
- Directory for storing experimental artifacts
- `logs/` - Model training logs and performance metrics
- `models/` - Saved model checkpoints and weights
- `samples/` - Molecule samples generated during training
- `inference/` - Molecules generated in inference mode
- `results/` - Experimental results and analyses

### Python scripts:
- `train.py` - Main script for training the DrugGEN model
- `inference.py` - Script for generating molecules using trained models
- `environment.yml` - Conda environment specification

&nbsp;
&nbsp;

## Datasets

The DrugGEN model requires two types of data for training: general compound data and target-specific bioactivity data. Both datasets were carefully curated to ensure high-quality training.

### Compound Data

The general compound dataset provides the model with knowledge about valid molecular structures and drug-like properties:

- **Source**: [ChEMBL v29 compound dataset](data/dataset_download.sh)
- **Size**: 1,588,865 stable organic molecules
- **Composition**: Molecules with a maximum of 45 atoms
- **Atom types**: C, O, N, F, Ca, K, Br, B, S, P, Cl, and As
- **Purpose**: Teaches the GAN module about valid chemical space and molecular structures

### Bioactivity Data

The target-specific dataset enables the model to learn the characteristics of molecules that interact with the selected protein target:

- **Target**: Human AKT1 protein (CHEMBL4282)
- **Sources**: 
  - ChEMBL bioactivity database (potent inhibitors with pChEMBL ≥ 6, equivalent to IC50 ≤ 1 µM)
  - DrugBank database (known AKT-interacting drug molecules)
- **Size**: [2,405 bioactive compounds](data/Filtered_AKT_inhibitors.csv)
- **Filtering**: Molecules larger than 45 heavy atoms were excluded
- **Purpose**: Guides the model to generate molecules with potential activity against AKT1

### Data Processing

Both datasets undergo extensive preprocessing to convert SMILES strings into graph representations suitable for the model. This includes:
- Conversion to molecular graphs
- Feature extraction and normalization
- Encoding of atom and bond types
- Size standardization

For more details on dataset construction and preprocessing methodology, please refer to our [paper](https://arxiv.org/abs/2302.07868).

<!-- ADD SOME INFO HERE -->

&nbsp;
&nbsp;

## Getting Started

### System Requirements

- **Operating System**: Ubuntu 20.04 or compatible Linux distribution
- **Python**: Version 3.9 or higher
- **Hardware**: 
  - CPU: Supports CPU-only operation
  - GPU: Recommended for faster training and inference (CUDA compatible)
- **RAM**: Minimum 8GB, 16GB+ recommended for larger datasets

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/HUBioDataLab/DrugGEN.git
   cd DrugGEN
   ```

2. **Set up the environment** (see [Training](#training) section for details):
   - Using conda: `conda env create -f environment.yml`

3. **Download datasets**:
   ```bash
   cd data
   bash dataset_download.sh
   cd ..
   ```

Now you're ready to start using DrugGEN for molecule generation or model training. Refer to the subsequent sections for specific usage instructions.

&nbsp;
&nbsp;

## Training

### Setting up environment

You can set up the environment using conda:

```bash
# Set up the environment (installs the requirements):
conda env create -f DrugGEN/environment.yml

# Activate the environment:
conda activate druggen
```

### Downloading input files

Before training, you need to download the necessary datasets:

```bash
# Navigate to the data directory
cd data

# Download dataset files
bash dataset_download.sh

# Return to the project root
cd ..
```

### Training the model

The default DrugGEN model can be trained with the following command:

```bash
python train.py --submodel="DrugGEN" \
                --raw_file="data/chembl_train.smi" \
                --dataset_file="chembl45_train.pt" \
                --drug_raw_file="data/akt_train.smi" \
                --drug_dataset_file="drugs_train.pt" \
                --max_atom=45
```

### Detailed Explanation of Arguments

Below is a comprehensive list of arguments that can be used to customize the training process:

#### Dataset Arguments
| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--raw_file` | SMILES containing text file for main dataset. Path to file. | `DrugGEN/data/chembl_train.smi` |
| `--drug_raw_file` | SMILES containing text file for target-specific dataset (e.g., AKT inhibitors). | `DrugGEN/data/akt_train.smi` |
| `--dataset_file` | Name for processed main dataset file to create or load. | `chembl45_train.pt` |
| `--drug_dataset_file` | Name for processed target-specific dataset file to create or load. | `drugs_train.pt` |
| `--mol_data_dir` | Directory where the dataset files are stored. | `DrugGEN/data` |
| `--drug_data_dir` | Directory where the drug dataset files are stored. | `DrugGEN/data` |
| `--features` | Whether to use additional node features (False uses atom types only). | `False` |

#### Model Arguments
| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--submodel` | Model variant to train: `DrugGEN` (target-specific) or `NoTarget` (non-target-specific). | `DrugGEN` |
| `--act` | Activation function for the model (`relu`, `tanh`, `leaky`, `sigmoid`). | `relu` |
| `--max_atom` | Maximum number of atoms in generated molecules. This is critical as the model uses one-shot generation. | `45` |
| `--dim` | Dimension of the Transformer Encoder model. Higher values increase model capacity but require more memory. | `128` |
| `--depth` | Depth (number of layers) of the Transformer model in generator. Deeper models can learn more complex features. | `1` |
| `--ddepth` | Depth of the Transformer model in discriminator. | `1` |
| `--heads` | Number of attention heads in the MultiHeadAttention module. | `8` |
| `--mlp_ratio` | MLP ratio for the Transformer, affects the feed-forward network size. | `3` |
| `--dropout` | Dropout rate for the generator encoder to prevent overfitting. | `0.0` |
| `--ddropout` | Dropout rate for the discriminator to prevent overfitting. | `0.0` |
| `--lambda_gp` | Gradient penalty lambda multiplier for Wasserstein GAN training stability. | `10` |

#### Training Arguments
| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--batch_size` | Number of molecules processed in each training batch. | `128` |
| `--epoch` | Total number of training epochs. | `10` |
| `--g_lr` | Learning rate for the Generator network. | `0.00001` |
| `--d_lr` | Learning rate for the Discriminator network. | `0.00001` |
| `--beta1` | Beta1 parameter for Adam optimizer, controls first moment decay. | `0.9` |
| `--beta2` | Beta2 parameter for Adam optimizer, controls second moment decay. | `0.999` |
| `--log_dir` | Directory to save training logs. | `DrugGEN/experiments/logs` |
| `--sample_dir` | Directory to save molecule samples during training. | `DrugGEN/experiments/samples` |
| `--model_save_dir` | Directory to save model checkpoints. | `DrugGEN/experiments/models` |
| `--log_sample_step` | Step interval for sampling and evaluating molecules during training. | `1000` |
| `--parallel` | Whether to parallelize training across multiple GPUs. | `False` |


#### Reproducibility Arguments
| Argument | Description | Default Value |
|----------|-------------|---------------|
| `--resume` | Whether to resume training from a checkpoint. | `False` |
| `--resume_epoch` | Epoch number to resume training from. | `None` |
| `--resume_iter` | Iteration step to resume training from. | `None` |
| `--resume_directory` | Directory containing model weights to load. | `None` |
| `--set_seed` | Whether to set a fixed random seed for reproducibility. | `False` |
| `--seed` | The random seed value to use if `set_seed` is True. | `1` |
| `--use_wandb` | Whether to use Weights & Biases for experiment tracking. | `False` |
| `--online` | Whether to use wandb in online mode (sync results during training). | `True` |
| `--exp_name` | Experiment name for wandb logging. | `druggen` |


&nbsp;
&nbsp;

## Molecule Generation Using Trained DrugGEN Models in the Inference Mode

- If you want to generate molecules using pre-trained models, it is recommended to use [Hugging Face](https://huggingface.co/spaces/HUBioDataLab/DrugGEN). Alternatively,

- First, download the weights of the chosen trained model from [trained models](https://drive.google.com/drive/folders/1biJLQeXCKqw4MzAYwOuJU6Aw5GIQlJMY), and place it in the folder: "DrugGEN/experiments/models/".
- After that, please run the code below:


```bash

python DrugGEN/inference.py --submodel="{Chosen model name}" --inference_model="DrugGEN/experiments/models/{Chosen model name}"
```

- SMILES representation of the generated molecules will be saved into the file: "DrugGEN/experiments/inference/{Chosen model name}/denovo_molecules.txt".

&nbsp;
&nbsp;

## Molecule Generation with Trained Models

### Using the Hugging Face Interface (Recommended)

For ease of use, we provide a [Hugging Face Space](https://huggingface.co/spaces/HUBioDataLab/DrugGEN) with a user-friendly interface for generating molecules using our pre-trained models.

### Local Generation Using Pre-trained Models

For local generation, follow these steps:

1. **Download pre-trained model weights**:
   Download the weights of your chosen model from our [model repository](https://drive.google.com/drive/folders/1biJLQeXCKqw4MzAYwOuJU6Aw5GIQlJMY)

2. **Place the model weights** in the `experiments/models/` directory

3. **Run inference**:
   ```bash
   python inference.py --submodel="[MODEL_NAME]" --inference_model="experiments/models/[MODEL_NAME]"
   ```
   Replace `[MODEL_NAME]` with the name of the specific model you downloaded.

4. **Output location**:
   The generated molecules in SMILES format will be saved to:
   ```
   experiments/inference/[MODEL_NAME]/denovo_molecules.txt
   ```

### Inference Parameters

You can customize the molecule generation process using the following parameters:

```bash
# Generate 100 molecules using the specified model with a batch size of 32
python inference.py --submodel="DrugGEN" \
                    --inference_model="experiments/models/DrugGEN" \
                    --n_samples=100 \
                    --batch_size=32
```

&nbsp;
&nbsp;

## Deep Learning-based Bioactivity Prediction

To evaluate the bioactivity of generated molecules against the AKT1 and CDK2 proteins, we utilize DEEPScreen, a deep learning-based virtual screening tool. Follow these steps to reproduce our bioactivity predictions:

### Setting up DEEPScreen

1. **Download the DEEPScreen model**:
   Download the pre-trained model from [this link](https://drive.google.com/file/d/1aG9oYspCsF9yG1gEGtFI_E2P4qlITqio/view?usp=drive_link)

2. **Extract the model files**:
   ```bash
   # Extract the downloaded file
   unzip DEEPScreen2.1.zip
   ```

### Running Predictions

Execute the following commands to predict bioactivity of your generated molecules:

```bash
# Navigate to the DEEPScreen directory
cd DEEPScreen2.1/chembl_31

# Run prediction for AKT target
python 8_Prediction.py AKT AKT
```

### Output

Prediction results will be saved in the following location:
```
DEEPScreen2.1/prediction_files/prediction_output/
```

These results include bioactivity scores that indicate the likelihood of interaction between the generated molecules and the AKT1 target protein. Higher scores suggest stronger potential binding affinity.

&nbsp;
&nbsp;

## Results (De Novo Generated Molecules of DrugGEN Models)

The system is trained to design effective inhibitory molecules against the AKT1 protein, which has critical importance for developing treatments against various types of cancer. SMILES notations of the de novo generated molecules from DrugGEN models, along with their deep learning-based bioactivity predictions (DeepScreen), docking and MD analyses, and filtering outcomes, can be accessed under the [paper_results](paper_results) folder. The structural representations of the final selected molecules are depicted in the figure below.

![structures](assets/Selected_denovo_AKT1_inhibitors.png)
**Fig. 2.** Promising de novo molecules to effectively target AKT1 protein (generated by DrugGEN model), selected via expert curation from the dataset of molecules with sufficiently low binding free energies (< -8 kcal/mol) in the molecular docking experiment.

&nbsp;
&nbsp;

## Updates

- 26/07/2024: DrugGEN pre-print is updated for v1.5 release.
- 04/06/2024: DrugGEN v1.5 is released.
- 30/01/2024: DrugGEN v1.0 is released.
- 15/02/2023: Our pre-print is shared [here](https://github.com/HUBioDataLab/DrugGEN/files/10828402/2302.07868.pdf).
- 01/01/2023: DrugGEN v0.1 is released.

&nbsp;
&nbsp;

## Citation
```bash
@misc{nl2023target,
    doi = {10.48550/ARXIV.2302.07868},
    title={Target Specific De Novo Design of Drug Candidate Molecules with Graph Transformer-based Generative Adversarial Networks},
    author={Atabey Ünlü and Elif Çevrim and Ahmet Sarıgün and Hayriye Çelikbilek and Heval Ataş Güvenilir and Altay Koyaş and Deniz Cansen Kahraman and Abdurrahman Olğaç and Ahmet Rifaioğlu and Tunca Doğan},
    year={2023},
    eprint={2302.07868},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Ünlü, A., Çevrim, E., Sarıgün, A., Yiğit, M.G., Çelikbilek, H., Bayram, O., Güvenilir, H.A., Koyaş, A., Kahraman, D.C., Olğaç, A., Rifaioğlu, A., Banoğlu, E., Doğan, T. (2023). Target Specific De Novo Design of Drug Candidate Molecules with Graph Transformer-based Generative Adversarial Networks. *arXiv preprint* arXiv:2302.07868.


&nbsp;
&nbsp;

## References/Resources

In each file, we indicate whether a function or script is imported from another source. Here are some excellent sources from which we benefit from: 
<!--ADD THE REFERENCES THAT WE USED DURING THE IMPLEMENTATION-->
- Molecule generation GAN schematic was inspired from [MolGAN](https://github.com/yongqyu/MolGAN-pytorch).
- [MOSES](https://github.com/molecularsets/moses) was used for performance calculation (MOSES Script are directly embedded to our code due to current installation issues related to the MOSES repo).
- [PyG](https://github.com/pyg-team/pytorch_geometric) was used to construct the custom dataset.
- Graph Transformer Encoder architecture was taken from [Dwivedi & Bresson (2021)](https://arxiv.org/abs/2012.09699) and [Vignac et al. (2022)](https://github.com/cvignac/DiGress) and modified. 

Our initial project repository was [this one](https://github.com/asarigun/DrugGEN).

&nbsp;
&nbsp;

## License
Copyright (C) 2024 HUBioDataLab

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.

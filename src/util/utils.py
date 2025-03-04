import os
import time
import math
import datetime
import warnings
import itertools
from copy import deepcopy
from functools import partial
from collections import Counter
from multiprocessing import Pool
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine as cos_distance

import torch
import wandb

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import (
    AllChem,
    Draw,
    Descriptors,
    Lipinski,
    Crippen,
    rdMolDescriptors,
    FilterCatalog,
)
from rdkit.Chem.Scaffolds import MurckoScaffold

# Disable RDKit warnings
RDLogger.DisableLog("rdApp.*")


class Metrics(object):
    """
    Collection of static methods to compute various metrics for molecules.
    """

    @staticmethod
    def valid(x):
        """
        Checks whether the molecule is valid.
        
        Args:
            x: RDKit molecule object.
        
        Returns:
            bool: True if molecule is valid and has a non-empty SMILES representation.
        """
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def tanimoto_sim_1v2(data1, data2):
        """
        Computes the average Tanimoto similarity for paired fingerprints.
        
        Args:
            data1: Fingerprint data for first set.
            data2: Fingerprint data for second set.
        
        Returns:
            float: The average Tanimoto similarity between corresponding fingerprints.
        """
        # Determine the minimum size between two arrays for pairing
        min_len = data1.size if data1.size > data2.size else data2
        sims = []
        for i in range(min_len):
            sim = DataStructs.FingerprintSimilarity(data1[i], data2[i])
            sims.append(sim)
        # Use 'mean' from statistics; note that variable 'sim' was used, corrected to use sims list.
        mean_sim = mean(sims)
        return mean_sim

    @staticmethod
    def mol_length(x):
        """
        Computes the length of the largest fragment (by character count) in a SMILES string.
        
        Args:
            x (str): SMILES string.
        
        Returns:
            int: Number of alphabetic characters in the longest fragment of the SMILES.
        """
        if x is not None:
            # Split at dots (.) and take the fragment with maximum length, then count alphabetic characters.
            return len([char for char in max(x.split(sep="."), key=len).upper() if char.isalpha()])
        else:
            return 0

    @staticmethod
    def max_component(data, max_len):
        """
        Returns the average normalized length of molecules in the dataset.
        
        Each molecule's length is computed and divided by max_len, then averaged.
        
        Args:
            data (iterable): Collection of SMILES strings.
            max_len (int): Maximum possible length for normalization.
        
        Returns:
            float: Normalized average length.
        """
        lengths = np.array(list(map(Metrics.mol_length, data)), dtype=np.float32)
        return (lengths / max_len).mean()

    @staticmethod
    def mean_atom_type(data):
        """
        Computes the average number of unique atom types in the provided node data.
        
        Args:
            data (iterable): Iterable containing node data with unique atom types.
        
        Returns:
            float: The average count of unique atom types, subtracting one.
        """
        atom_types_used = []
        for i in data:
            # Assuming each element i has a .unique() method that returns unique atom types.
            atom_types_used.append(len(i.unique().tolist()))
        av_type = np.mean(atom_types_used) - 1
        return av_type


def mols2grid_image(mols, path):
    """
    Saves grid images for a list of molecules.
    
    For each molecule in the list, computes 2D coordinates and saves an image file.
    
    Args:
        mols (list): List of RDKit molecule objects.
        path (str): Directory where images will be saved.
    """
    # Replace None molecules with an empty molecule
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for i in range(len(mols)):
        if Metrics.valid(mols[i]):
            AllChem.Compute2DCoords(mols[i])
            file_path = os.path.join(path, "{}.png".format(i + 1))
            Draw.MolToFile(mols[i], file_path, size=(1200, 1200))
            # wandb.save(file_path)  # Optionally save to Weights & Biases
        else:
            continue


def save_smiles_matrices(mols, edges_hard, nodes_hard, path, data_source=None):
    """
    Saves the edge and node matrices along with SMILES strings to text files.
    
    Each file contains the edge matrix, node matrix, and SMILES representation for a molecule.
    
    Args:
        mols (list): List of RDKit molecule objects.
        edges_hard (torch.Tensor): Tensor of edge features.
        nodes_hard (torch.Tensor): Tensor of node features.
        path (str): Directory where files will be saved.
        data_source: Optional data source information (not used in function).
    """
    mols = [e if e is not None else Chem.RWMol() for e in mols]

    for i in range(len(mols)):
        if Metrics.valid(mols[i]):
            save_path = os.path.join(path, "{}.txt".format(i + 1))
            with open(save_path, "a") as f:
                np.savetxt(f, edges_hard[i].cpu().numpy(), header="edge matrix:\n", fmt='%1.2f')
                f.write("\n")
                np.savetxt(f, nodes_hard[i].cpu().numpy(), header="node matrix:\n", footer="\nsmiles:", fmt='%1.2f')
                f.write("\n")
            # Append the SMILES representation to the file
            with open(save_path, "a") as f:
                print(Chem.MolToSmiles(mols[i]), file=f)
            # wandb.save(save_path)  # Optionally save to Weights & Biases
        else:
            continue

def dense_to_sparse_with_attr(adj):
    """
    Converts a dense adjacency matrix to a sparse representation.
    
    Args:
        adj (torch.Tensor): Adjacency matrix tensor (2D or 3D) with square last two dimensions.
    
    Returns:
        tuple: A tuple containing indices and corresponding edge attributes.
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
    return index, edge_attr


def mol_sample(sample_directory, edges, nodes, idx, i, matrices2mol, dataset_name):
    """
    Samples molecules from edge and node predictions, then saves grid images and text files.
    
    Args:
        sample_directory (str): Directory to save the samples.
        edges (torch.Tensor): Edge predictions tensor.
        nodes (torch.Tensor): Node predictions tensor.
        idx (int): Current index for naming the sample.
        i (int): Epoch/iteration index.
        matrices2mol (callable): Function to convert matrices to RDKit molecule.
        dataset_name (str): Name of the dataset for file naming.
    """
    sample_path = os.path.join(sample_directory, "{}_{}-epoch_iteration".format(idx + 1, i + 1))
    # Get the index of the maximum predicted feature along the last dimension
    g_edges_hat_sample = torch.max(edges, -1)[1]
    g_nodes_hat_sample = torch.max(nodes, -1)[1]
    # Convert matrices to molecule objects
    mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                        strict=True, file_name=dataset_name)
           for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    mols2grid_image(mol, sample_path)
    save_smiles_matrices(mol, g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), sample_path)

    # Remove the directory if no files were saved
    if len(os.listdir(sample_path)) == 0:
        os.rmdir(sample_path)

    print("Valid molecules are saved.")
    print("Valid matrices and smiles are saved")


def logging(log_path, start_time, i, idx, loss, save_path, drug_smiles, edge, node, 
            matrices2mol, dataset_name, real_adj, real_annot, drug_vecs):
    """
    Logs training statistics and evaluation metrics.
    
    The function generates molecules from predictions, computes various metrics such as
    validity, uniqueness, novelty, and similarity scores, and logs them using wandb and a file.
    
    Args:
        log_path (str): Path to save the log file.
        start_time (float): Start time to compute elapsed time.
        i (int): Current iteration index.
        idx (int): Current epoch index.
        loss (dict): Dictionary to update with loss and metric values.
        save_path (str): Directory path to save sample outputs.
        drug_smiles (list): List of reference drug SMILES.
        edge (torch.Tensor): Edge prediction tensor.
        node (torch.Tensor): Node prediction tensor.
        matrices2mol (callable): Function to convert matrices to molecules.
        dataset_name (str): Dataset name.
        real_adj (torch.Tensor): Ground truth adjacency matrix tensor.
        real_annot (torch.Tensor): Ground truth annotation tensor.
        drug_vecs (list): List of drug vectors for similarity calculation.
    """
    g_edges_hat_sample = torch.max(edge, -1)[1]
    g_nodes_hat_sample = torch.max(node, -1)[1]

    a_tensor_sample = torch.max(real_adj, -1)[1].float()
    x_tensor_sample = torch.max(real_annot, -1)[1].float()

    # Generate molecules from predictions and real data
    mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                         strict=True, file_name=dataset_name)
            for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]
    real_mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(),
                              strict=True, file_name=dataset_name)
                for e_, n_ in zip(a_tensor_sample, x_tensor_sample)]

    # Compute average number of atom types
    atom_types_average = Metrics.mean_atom_type(g_nodes_hat_sample)
    real_smiles = [Chem.MolToSmiles(x) for x in real_mol if x is not None]
    gen_smiles = []
    uniq_smiles = []
    for line in mols:
        if line is not None:
            gen_smiles.append(Chem.MolToSmiles(line))
            uniq_smiles.append(Chem.MolToSmiles(line))
        elif line is None:
            gen_smiles.append(None)

    # Process SMILES to take the longest fragment if multiple are present
    gen_smiles_saves = [None if x is None else max(x.split('.'), key=len) for x in gen_smiles]
    uniq_smiles_saves = [None if x is None else max(x.split('.'), key=len) for x in uniq_smiles]

    # Save the generated SMILES to a text file
    sample_save_dir = os.path.join(save_path, "samples.txt")
    with open(sample_save_dir, "a") as f:
        for s in gen_smiles_saves:
            if s is not None:
                f.write(s + "\n")

    k = len(set(uniq_smiles_saves) - {None})
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log_str = "Elapsed [{}], Epoch/Iteration [{}/{}]".format(et, idx, i + 1)
    
    # Generate molecular fingerprints for similarity computations
    gen_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols if x is not None]
    chembl_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in real_mol if x is not None]

    # Compute evaluation metrics: validity, uniqueness, novelty, similarity scores, and average maximum molecule length.
    valid = fraction_valid(gen_smiles_saves)
    unique = fraction_unique(uniq_smiles_saves, k, check_validity=False)
    novel_starting_mol = novelty(gen_smiles_saves, real_smiles)
    novel_akt = novelty(gen_smiles_saves, drug_smiles)
    if len(uniq_smiles_saves) == 0:
        snn_chembl = 0
        snn_akt = 0
        maxlen = 0
    else:
        snn_chembl = average_agg_tanimoto(np.array(chembl_vecs), np.array(gen_vecs))
        snn_akt = average_agg_tanimoto(np.array(drug_vecs), np.array(gen_vecs))
        maxlen = Metrics.max_component(uniq_smiles_saves, 45)

    # Update loss dictionary with computed metrics
    loss.update({
        'Validity': valid,
        'Uniqueness': unique,
        'Novelty': novel_starting_mol,
        'Novelty_akt': novel_akt,
        'SNN_chembl': snn_chembl,
        'SNN_akt': snn_akt,
        'MaxLen': maxlen,
        'Atom_types': atom_types_average
    })

    # Log metrics using wandb
    wandb.log({
        "Validity": valid,
        "Uniqueness": unique,
        "Novelty": novel_starting_mol,
        "Novelty_akt": novel_akt,
        "SNN_chembl": snn_chembl,
        "SNN_akt": snn_akt,
        "MaxLen": maxlen,
        "Atom_types": atom_types_average
    })

    # Append each metric to the log string and write to the log file
    for tag, value in loss.items():
        log_str += ", {}: {:.4f}".format(tag, value)
    with open(log_path, "a") as f:
        f.write(log_str + "\n")
    print(log_str)
    print("\n")


def plot_grad_flow(named_parameters, model, itera, epoch, grad_flow_directory):
    """
    Plots the gradients flowing through different layers during training.
    
    This is useful to check for possible gradient vanishing or exploding problems.
    
    Args:
        named_parameters (iterable): Iterable of (name, parameter) tuples from the model.
        model (str): Name of the model (used for saving the plot).
        itera (int): Iteration index.
        epoch (int): Current epoch.
        grad_flow_directory (str): Directory to save the gradient flow plot.
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    # Plot maximum gradients and average gradients for each layer
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=1)  # Zoom in on lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([
        Line2D([0], [0], color="c", lw=4),
        Line2D([0], [0], color="b", lw=4),
        Line2D([0], [0], color="k", lw=4)
    ], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    # Save the plot to the specified directory
    plt.savefig(os.path.join(grad_flow_directory, "weights_" + model + "_" + str(itera) + "_" + str(epoch) + ".png"), dpi=500, bbox_inches='tight')


def get_mol(smiles_or_mol):
    """
    Loads a SMILES string or molecule into an RDKit molecule object.
    
    Args:
        smiles_or_mol (str or RDKit Mol): SMILES string or RDKit molecule.
    
    Returns:
        RDKit Mol or None: Sanitized molecule object, or None if invalid.
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def mapper(n_jobs):
    """
    Returns a mapping function for parallel or serial processing.
    
    If n_jobs == 1, returns the built-in map function.
    If n_jobs > 1, returns a function that uses a multiprocessing pool.
    
    Args:
        n_jobs (int or pool object): Number of jobs or a Pool instance.
    
    Returns:
        callable: A function that acts like map.
    """
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))
        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)
        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result
        return _mapper
    return n_jobs.map


def remove_invalid(gen, canonize=True, n_jobs=1):
    """
    Removes invalid molecules from the provided dataset.
    
    Optionally canonizes the SMILES strings.
    
    Args:
        gen (list): List of SMILES strings.
        canonize (bool): Whether to convert to canonical SMILES.
        n_jobs (int): Number of parallel jobs.
    
    Returns:
        list: Filtered list of valid molecules.
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if x is not None]


def fraction_valid(gen, n_jobs=1):
    """
    Computes the fraction of valid molecules in the dataset.
    
    Args:
        gen (list): List of SMILES strings.
        n_jobs (int): Number of parallel jobs.
    
    Returns:
        float: Fraction of molecules that are valid.
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)


def canonic_smiles(smiles_or_mol):
    """
    Converts a SMILES string or molecule to its canonical SMILES.
    
    Args:
        smiles_or_mol (str or RDKit Mol): Input molecule.
    
    Returns:
        str or None: Canonical SMILES string or None if invalid.
    """
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
    """
    Computes the fraction of unique molecules.
    
    Optionally computes unique@k, where only the first k molecules are considered.
    
    Args:
        gen (list): List of SMILES strings.
        k (int): Optional cutoff for unique@k computation.
        n_jobs (int): Number of parallel jobs.
        check_validity (bool): Whether to check for validity of molecules.
    
    Returns:
        float: Fraction of unique molecules.
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn("Can't compute unique@{}.".format(k) +
                          " gen contains only {} molecules".format(len(gen)))
        gen = gen[:k]
    if check_validity:
        canonic = list(mapper(n_jobs)(canonic_smiles, gen))
        canonic = [i for i in canonic if i is not None]
    set_cannonic = set(canonic)
    return 0 if len(canonic) == 0 else len(set_cannonic) / len(canonic)


def novelty(gen, train, n_jobs=1):
    """
    Computes the novelty score of generated molecules.
    
    Novelty is defined as the fraction of generated molecules that do not appear in the training set.
    
    Args:
        gen (list): List of generated SMILES strings.
        train (list): List of training SMILES strings.
        n_jobs (int): Number of parallel jobs.
    
    Returns:
        float: Novelty score.
    """
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return 0 if len(gen_smiles_set) == 0 else len(gen_smiles_set - train_set) / len(gen_smiles_set)


def internal_diversity(gen):
    """
    Computes the internal diversity of a set of molecules.
    
    Internal diversity is defined as one minus the average Tanimoto similarity between all pairs.
    
    Args:
        gen: Array-like representation of molecules.
    
    Returns:
        tuple: Mean and standard deviation of internal diversity.
    """
    diversity = [1 - x for x in average_agg_tanimoto(gen, gen, agg="mean", intdiv=True)]
    return np.mean(diversity), np.std(diversity)


def average_agg_tanimoto(stock_vecs, gen_vecs, batch_size=5000, agg='max', device='cpu', p=1, intdiv=False):
    """
    Computes the average aggregated Tanimoto similarity between two sets of molecular fingerprints.
    
    For each fingerprint in gen_vecs, finds the closest (max or mean) similarity with fingerprints in stock_vecs.
    
    Args:
        stock_vecs (numpy.ndarray): Array of fingerprint vectors from the reference set.
        gen_vecs (numpy.ndarray): Array of fingerprint vectors from the generated set.
        batch_size (int): Batch size for processing fingerprints.
        agg (str): Aggregation method, either 'max' or 'mean'.
        device (str): Device to perform computations on.
        p (int): Power for averaging.
        intdiv (bool): Whether to return individual similarities or the average.
    
    Returns:
        float or numpy.ndarray: Average aggregated Tanimoto similarity or array of individual scores.
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            # Compute Jaccard/Tanimoto similarity
            jac = (tp / (x_stock.sum(1, keepdim=True) + y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac ** p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto) ** (1 / p)
    if intdiv:
        return agg_tanimoto
    else:
        return np.mean(agg_tanimoto)


def str2bool(v):
    """
    Converts a string to a boolean.
    
    Args:
        v (str): Input string.
    
    Returns:
        bool: True if the string is 'true' (case insensitive), else False.
    """
    return v.lower() in ('true')


def obey_lipinski(mol):
    """
    Checks if a molecule obeys Lipinski's Rule of Five.
    
    The function evaluates weight, hydrogen bond donors and acceptors, logP, and rotatable bonds.
    
    Args:
        mol (RDKit Mol): Molecule object.
    
    Returns:
        int: Number of Lipinski rules satisfied.
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp := Crippen.MolLogP(mol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def obey_veber(mol):
    """
    Checks if a molecule obeys Veber's rules.
    
    Veber's rules focus on the number of rotatable bonds and topological polar surface area.
    
    Args:
        mol (RDKit Mol): Molecule object.
    
    Returns:
        int: Number of Veber's rules satisfied.
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    rule_2 = rdMolDescriptors.CalcTPSA(mol) <= 140
    return np.sum([int(a) for a in [rule_1, rule_2]])


def load_pains_filters():
    """
    Loads the PAINS (Pan-Assay INterference compoundS) filters A, B, and C.
    
    Returns:
        FilterCatalog: An RDKit FilterCatalog object containing PAINS filters.
    """
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)
    return catalog


def is_pains(mol, catalog):
    """
    Checks if the given molecule is a PAINS compound.
    
    Args:
        mol (RDKit Mol): Molecule object.
        catalog (FilterCatalog): A catalog of PAINS filters.
    
    Returns:
        bool: True if the molecule matches a PAINS filter, else False.
    """
    entry = catalog.GetFirstMatch(mol)
    return entry is not None


def mapper(n_jobs):
    """
    Returns a mapping function for parallel or serial processing.
    
    If n_jobs == 1, returns the built-in map function.
    If n_jobs > 1, returns a function that uses a multiprocessing pool.
    
    Args:
        n_jobs (int or pool object): Number of jobs or a Pool instance.
    
    Returns:
        callable: A function that acts like map.
    """
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))
        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)
        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result
        return _mapper
    return n_jobs.map


def fragmenter(mol):
    """
    Fragments a molecule using BRICS and returns a list of fragment SMILES.
    
    Args:
        mol (str or RDKit Mol): Input molecule.
    
    Returns:
        list: List of fragment SMILES strings.
    """
    fgs = AllChem.FragmentOnBRICSBonds(get_mol(mol))
    fgs_smi = Chem.MolToSmiles(fgs).split(".")
    return fgs_smi


def get_mol(smiles_or_mol):
    """
    Loads a SMILES string or molecule into an RDKit molecule object.
    
    Args:
        smiles_or_mol (str or RDKit Mol): SMILES string or molecule.
    
    Returns:
        RDKit Mol or None: Sanitized molecule object or None if invalid.
    """
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol


def compute_fragments(mol_list, n_jobs=1):
    """
    Fragments a list of molecules using BRICS and returns a counter of fragment occurrences.
    
    Args:
        mol_list (list): List of molecules (SMILES or RDKit Mol).
        n_jobs (int): Number of parallel jobs.
    
    Returns:
        Counter: A Counter dictionary mapping fragment SMILES to counts.
    """
    fragments = Counter()
    for mol_frag in mapper(n_jobs)(fragmenter, mol_list):
        fragments.update(mol_frag)
    return fragments


def compute_scaffolds(mol_list, n_jobs=1, min_rings=2):
    """
    Extracts scaffolds from a list of molecules as canonical SMILES.
    
    Only scaffolds with at least min_rings rings are considered.
    
    Args:
        mol_list (list): List of molecules.
        n_jobs (int): Number of parallel jobs.
        min_rings (int): Minimum number of rings required in a scaffold.
    
    Returns:
        Counter: A Counter mapping scaffold SMILES to counts.
    """
    scaffolds = Counter()
    map_ = mapper(n_jobs)
    scaffolds = Counter(map_(partial(compute_scaffold, min_rings=min_rings), mol_list))
    if None in scaffolds:
        scaffolds.pop(None)
    return scaffolds


def get_n_rings(mol):
    """
    Computes the number of rings in a molecule.
    
    Args:
        mol (RDKit Mol): Molecule object.
    
    Returns:
        int: Number of rings.
    """
    return mol.GetRingInfo().NumRings()


def compute_scaffold(mol, min_rings=2):
    """
    Computes the Murcko scaffold of a molecule and returns its canonical SMILES if it has enough rings.
    
    Args:
        mol (str or RDKit Mol): Input molecule.
        min_rings (int): Minimum number of rings required.
    
    Returns:
        str or None: Canonical SMILES of the scaffold if valid, else None.
    """
    mol = get_mol(mol)
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except (ValueError, RuntimeError):
        return None
    n_rings = get_n_rings(scaffold)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    if scaffold_smiles == '' or n_rings < min_rings:
        return None
    return scaffold_smiles


class Metric:
    """
    Abstract base class for chemical metrics.
    
    Derived classes should implement the precalc and metric methods.
    """
    def __init__(self, n_jobs=1, device='cpu', batch_size=512, **kwargs):
        self.n_jobs = n_jobs
        self.device = device
        self.batch_size = batch_size
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __call__(self, ref=None, gen=None, pref=None, pgen=None):
        """
        Computes the metric between reference and generated molecules.
        
        Exactly one of ref or pref, and gen or pgen should be provided.
        
        Args:
            ref: Reference molecule list.
            gen: Generated molecule list.
            pref: Precalculated reference metric.
            pgen: Precalculated generated metric.
        
        Returns:
            Metric value computed by the metric method.
        """
        assert (ref is None) != (pref is None), "specify ref xor pref"
        assert (gen is None) != (pgen is None), "specify gen xor pgen"
        if pref is None:
            pref = self.precalc(ref)
        if pgen is None:
            pgen = self.precalc(gen)
        return self.metric(pref, pgen)

    def precalc(self, molecules):
        """
        Pre-calculates necessary representations from a list of molecules.
        Should be implemented by derived classes.
        """
        raise NotImplementedError

    def metric(self, pref, pgen):
        """
        Computes the metric given precalculated representations.
        Should be implemented by derived classes.
        """
        raise NotImplementedError


class FragMetric(Metric):
    """
    Metrics based on molecular fragments.
    """
    def precalc(self, mols):
        return {'frag': compute_fragments(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['frag'], pgen['frag'])


class ScafMetric(Metric):
    """
    Metrics based on molecular scaffolds.
    """
    def precalc(self, mols):
        return {'scaf': compute_scaffolds(mols, n_jobs=self.n_jobs)}

    def metric(self, pref, pgen):
        return cos_similarity(pref['scaf'], pgen['scaf'])


def cos_similarity(ref_counts, gen_counts):
    """
    Computes cosine similarity between two molecular vectors.
    
    Args:
        ref_counts (dict): Reference molecular vectors.
        gen_counts (dict): Generated molecular vectors.
    
    Returns:
        float: Cosine similarity between the two molecular vectors.
    """
    if len(ref_counts) == 0 or len(gen_counts) == 0:
        return np.nan
    keys = np.unique(list(ref_counts.keys()) + list(gen_counts.keys()))
    ref_vec = np.array([ref_counts.get(k, 0) for k in keys])
    gen_vec = np.array([gen_counts.get(k, 0) for k in keys])
    return 1 - cos_distance(ref_vec, gen_vec)
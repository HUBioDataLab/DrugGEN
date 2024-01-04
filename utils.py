from statistics import mean
import os
import math
import time
import datetime
from rdkit import DataStructs
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch 
import wandb
RDLogger.DisableLog('rdApp.*')
import warnings
from multiprocessing import Pool
class Metrics(object):

    @staticmethod
    def valid(x):
        return x is not None and Chem.MolToSmiles(x) != ''

    @staticmethod
    def tanimoto_sim_1v2(data1, data2):
        min_len = data1.size if data1.size > data2.size else data2
        sims = []
        for i in range(min_len):
            sim = DataStructs.FingerprintSimilarity(data1[i], data2[i])
            sims.append(sim)
        mean_sim = mean(sim)
        return mean_sim

    @staticmethod
    def mol_length(x):
        if x is not None:
            return  len([char for char in max(x.split(sep =".")).upper() if char.isalpha()])
        else:
            return 0
    
    @staticmethod
    def max_component(data, max_len):

        # There will be a name change for this function to better reflect what it does
        
        """Returns the average length of the molecules in the dataset normalized by the maximum length.

        Returns:
            array: normalized average length of the molecules in the dataset
        """

        return ((np.array(list(map(Metrics.mol_length, data)), dtype=np.float32)/max_len).mean())

    @staticmethod
    def mean_atom_type(data):
        atom_types_used = []
        for i in data:

            atom_types_used.append(len(i.unique().tolist()))
        av_type = np.mean(atom_types_used) - 1
        
        return av_type


def sim_reward(mol_gen, fps_r):
    
    gen_scaf = []
    
    for x in mol_gen: 
        if x is not None:
            try:
                
                gen_scaf.append(MurckoScaffold.GetScaffoldForMol(x))
            except:
                pass

    if len(gen_scaf) == 0:
        
        rew = 1
    else:
        fps = [Chem.RDKFingerprint(x) for x in gen_scaf]
            
        
        fps = np.array(fps)
        fps_r = np.array(fps_r)
    
        rew =  average_agg_tanimoto(fps_r,fps)
        if math.isnan(rew):
            rew = 1
    
    return rew  ## change this to penalty

##########################################
##########################################
##########################################

def mols2grid_image(mols,path):
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    
    for i in range(len(mols)):
        if Metrics.valid(mols[i]):
            AllChem.Compute2DCoords(mols[i])
            Draw.MolToFile(mols[i], os.path.join(path,"{}.png".format(i+1)), size=(1200,1200))
            #wandb.save(os.path.join(path,"{}.png".format(i+1)))
        else:
            continue

def save_smiles_matrices(mols,edges_hard, nodes_hard, path, data_source = None): 
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    
    for i in range(len(mols)):
        if Metrics.valid(mols[i]):
            save_path = os.path.join(path,"{}.txt".format(i+1))
            with open(save_path, "a") as f:
                np.savetxt(f, edges_hard[i].cpu().numpy(), header="edge matrix:\n",fmt='%1.2f')
                f.write("\n")
                np.savetxt(f, nodes_hard[i].cpu().numpy(), header="node matrix:\n", footer="\nsmiles:",fmt='%1.2f')
                f.write("\n")
                #f.write(m0)
                f.write("\n")
            print(Chem.MolToSmiles(mols[i]), file=open(save_path,"a"))
            #wandb.save(save_path)
        else:
            continue


##########################################
##########################################
##########################################


def dense_to_sparse_with_attr(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])
        #index = torch.stack(index, dim=0)
    return index, edge_attr


def label2onehot(labels, dim, device):
    """Convert label indices to one-hot vectors."""
    out = torch.zeros(list(labels.size())+[dim]).to(device)
    out.scatter_(len(out.size())-1,labels.unsqueeze(-1),1.)

    return out.float()


def mol_sample(sample_directory, edges, nodes, idx, i,matrices2mol, dataset_name):
    sample_path = os.path.join(sample_directory,"{}_{}-epoch_iteration".format(idx+1, i+1))
    g_edges_hat_sample = torch.max(edges, -1)[1]
    g_nodes_hat_sample = torch.max(nodes , -1)[1]
    mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=dataset_name)
            for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]

    if not os.path.exists(sample_path):
        os.makedirs(sample_path)

    mols2grid_image(mol,sample_path)
    save_smiles_matrices(mol,g_edges_hat_sample.detach(), g_nodes_hat_sample.detach(), sample_path)

    if len(os.listdir(sample_path)) == 0:
        os.rmdir(sample_path)

    print("Valid molecules are saved.")
    print("Valid matrices and smiles are saved")


def logging(log_path, start_time, i, idx, loss, save_path, drug_smiles, edge, node, 
            matrices2mol, dataset_name, real_adj, real_annot, drug_vecs):

    g_edges_hat_sample = torch.max(edge, -1)[1]
    g_nodes_hat_sample = torch.max(node , -1)[1]

    a_tensor_sample = torch.max(real_adj, -1)[1].float()
    x_tensor_sample = torch.max(real_annot, -1)[1].float()

    mols = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=dataset_name)
            for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]

    real_mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=dataset_name)
            for e_, n_ in zip(a_tensor_sample, x_tensor_sample)]

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

    gen_smiles_saves = [None if x is None else max(x.split('.'), key=len) for x in gen_smiles]
    uniq_smiles_saves = [None if x is None else max(x.split('.'), key=len) for x in uniq_smiles]

    sample_save_dir = os.path.join(save_path, "samples.txt")
    with open(sample_save_dir, "a") as f:
        for idxs in range(len(gen_smiles_saves)):
            if gen_smiles_saves[idxs] is not None:
                f.write(gen_smiles_saves[idxs])
                f.write("\n")

    k = len(set(uniq_smiles_saves) - {None})
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Elapsed [{}], Epoch/Iteration [{}/{}]".format(et, idx, i+1)
    gen_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in mols if x is not None]
    chembl_vecs = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in real_mol if x is not None]

    # Log update
    #m0 = get_all_metrics(gen = gen_smiles, train = train_smiles, batch_size=batch_size, k = valid_mol_num, device=self.device)
    valid = fraction_valid(gen_smiles_saves)
    unique = fraction_unique(uniq_smiles_saves, k, check_validity=False)
    novel_starting_mol = novelty(gen_smiles_saves, real_smiles)
    novel_akt = novelty(gen_smiles_saves, drug_smiles)
    if (len(uniq_smiles_saves) == 0):
        snn_chembl = 0
        snn_akt = 0
        maxlen = 0
    else:
        snn_chembl = average_agg_tanimoto(np.array(chembl_vecs),np.array(gen_vecs))
        snn_akt = average_agg_tanimoto(np.array(drug_vecs),np.array(gen_vecs))
        maxlen = Metrics.max_component(uniq_smiles_saves, 45)

    loss.update({'Validity': valid})
    loss.update({'Uniqueness': unique})
    loss.update({'Novelty': novel_starting_mol})
    loss.update({'Novelty_akt': novel_akt})
    loss.update({'SNN_chembl': snn_chembl})
    loss.update({'SNN_akt': snn_akt})
    loss.update({'MaxLen': maxlen})
    loss.update({'Atom_types': atom_types_average})

    wandb.log({"Validity": valid, "Uniqueness": unique, "Novelty": novel_starting_mol,
                "Novelty_akt": novel_akt, "SNN_chembl": snn_chembl, "SNN_akt": snn_akt,
                  "MaxLen": maxlen, "Atom_types": atom_types_average})

    for tag, value in loss.items():
        log += ", {}: {:.4f}".format(tag, value)
    with open(log_path, "a") as f:
        f.write(log)
        f.write("\n")
    print(log)
    print("\n")


def plot_grad_flow(named_parameters, model, itera, epoch,grad_flow_directory):
    # Based on https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            #print(p.grad,n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=1) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    pltsavedir = grad_flow_directory
    plt.savefig(os.path.join(pltsavedir, "weights_" + model  + "_"  + str(itera) + "_" + str(epoch) +  ".png"), dpi= 500,bbox_inches='tight')


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
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
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
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
    Removes invalid molecules from the dataset
    """
    if not canonize:
        mols = mapper(n_jobs)(get_mol, gen)
        return [gen_ for gen_, mol in zip(gen, mols) if mol is not None]
    return [x for x in mapper(n_jobs)(canonic_smiles, gen) if
            x is not None]


def fraction_valid(gen, n_jobs=1):
    """
    Computes a number of valid molecules
    Parameters:
        gen: list of SMILES
        n_jobs: number of threads for calculation
    """
    gen = mapper(n_jobs)(get_mol, gen)
    return 1 - gen.count(None) / len(gen)
def canonic_smiles(smiles_or_mol):
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)
def fraction_unique(gen, k=None, n_jobs=1, check_validity=False):
    """
    Computes a number of unique molecules
    Parameters:
        gen: list of SMILES
        k: compute unique@k
        n_jobs: number of threads for calculation
        check_validity: raises ValueError if invalid molecules are present
    """
    if k is not None:
        if len(gen) < k:
            warnings.warn(
                "Can't compute unique@{}.".format(k) +
                "gen contains only {} molecules".format(len(gen))
            )
        gen = gen[:k]
    canonic = set(mapper(n_jobs)(canonic_smiles, gen))
    if None in canonic and check_validity:
        #canonic = [i for i in canonic if i is not None]
        raise ValueError("Invalid molecule passed to unique@k")
    return 0 if len(gen) == 0 else len(canonic) / len(gen)

def novelty(gen, train, n_jobs=1):
    gen_smiles = mapper(n_jobs)(canonic_smiles, gen)
    gen_smiles_set = set(gen_smiles) - {None}
    train_set = set(train)
    return 0 if len(gen_smiles_set) == 0 else len(gen_smiles_set - train_set) / len(gen_smiles_set)



def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
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
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

def str2bool(v):
    return v.lower() in ('true')
from statistics import mean
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rdkit import RDLogger  
import torch 
from rdkit.Chem.Scaffolds import MurckoScaffold
import math
import time
import datetime
import re
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
            return  len([char for char in max(Chem.MolToSmiles(x).split(sep =".")).upper() if char.isalpha()])
        else:
            return 0
    
    @staticmethod
    def max_component(data, max_len):
        
        return (np.array(list(map(Metrics.mol_length, data)), dtype=np.float32)/max_len).mean()       

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
    
        rew =  average_agg_tanimoto(fps_r,fps)[0]
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
        #if Chem.MolToSmiles(mols[i]) != '':
            AllChem.Compute2DCoords(mols[i])
            Draw.MolToFile(mols[i], os.path.join(path,"{}.png".format(i+1)), size=(1200,1200)) 
        else:
            continue

def save_smiles_matrices(mols,edges_hard, nodes_hard,path,data_source = None): 
    mols = [e if e is not None else Chem.RWMol() for e in mols]
    
    for i in range(len(mols)):
        if Metrics.valid(mols[i]):
            #m0= all_scores_for_print(mols[i], data_source, norm=False)
        #if Chem.MolToSmiles(mols[i]) != '':
            save_path = os.path.join(path,"{}.txt".format(i+1))
            with open(save_path, "a") as f:
                np.savetxt(f, edges_hard[i].cpu().numpy(), header="edge matrix:\n",fmt='%1.2f')
                f.write("\n")
                np.savetxt(f, nodes_hard[i].cpu().numpy(), header="node matrix:\n", footer="\nsmiles:",fmt='%1.2f')
                f.write("\n")
                #f.write(m0)
                f.write("\n")
        

            print(Chem.MolToSmiles(mols[i]), file=open(save_path,"a"))
        else:
            continue

##########################################
##########################################
##########################################
  
def dense_to_sparse_with_attr(adj):
    ### 
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


def sample_z_node(batch_size, vertexes, nodes):
    
    ''' Random noise for nodes logits. '''
    
    return np.random.normal(0,1, size=(batch_size,vertexes, nodes))  #  128, 9, 5


def sample_z_edge(batch_size, vertexes, edges):
    
    ''' Random noise for edges logits. '''
    
    return np.random.normal(0,1, size=(batch_size, vertexes, vertexes, edges)) # 128, 9, 9, 5

def sample_z( batch_size, z_dim):
    
    ''' Random noise. '''
    
    return np.random.normal(0,1, size=(batch_size,z_dim))  #  128, 9, 5       


def mol_sample(sample_directory, model_name, mol, edges, nodes, idx, i):
    sample_path = os.path.join(sample_directory,"{}-{}_{}-epoch_iteration".format(model_name,idx+1, i+1))
    
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
        
    mols2grid_image(mol,sample_path)
    
    save_smiles_matrices(mol,edges.detach(), nodes.detach(), sample_path)
    
    if len(os.listdir(sample_path)) == 0:
        os.rmdir(sample_path)
                            
    print("Valid molecules are saved.")
    print("Valid matrices and smiles are saved")




    
def logging(log_path, start_time, mols, train_smiles, i,idx, loss,model_num, save_path):
    
    gen_smiles =  []    
    for line in mols:
        if line is not None:
            gen_smiles.append(Chem.MolToSmiles(line))
        elif line is None:
            gen_smiles.append(None)

    #gen_smiles_saves = [None if x is None else re.sub('\*', '', x) for x in gen_smiles]
    #gen_smiles_saves = [None if x is None else re.sub('\.', '', x) for x in gen_smiles_saves]
    gen_smiles_saves = [None if x is None else max(x.split('.'), key=len) for x in gen_smiles]

    sample_save_dir = os.path.join(save_path, "samples-GAN{}.txt".format(model_num))
    with open(sample_save_dir, "a") as f:
        for idxs in range(len(gen_smiles_saves)):
            if gen_smiles_saves[idxs] is not None:
                
                f.write(gen_smiles_saves[idxs])
                f.write("\n")

    k = len(set(gen_smiles_saves) - {None})  
                        
                              
    et = time.time() - start_time
    et = str(datetime.timedelta(seconds=et))[:-7]
    log = "Elapsed [{}], Epoch/Iteration [{}/{}] for GAN{}".format(et, idx,  i+1, model_num)
    
    # Log update
    #m0 = get_all_metrics(gen = gen_smiles, train = train_smiles, batch_size=batch_size, k = valid_mol_num, device=self.device)
    valid = fraction_valid(gen_smiles_saves)
    unique = fraction_unique(gen_smiles_saves, k, check_validity=False)
    novel = novelty(gen_smiles_saves, train_smiles)
    
    #qed = [QED(mol) for mol in mols if mol is not None]
    #sa = [SA(mol) for mol in mols if mol is not None]
    #logp = [logP(mol) for mol in mols if mol is not None]
    
    #IntDiv = internal_diversity(gen_smiles) 
    #m0= all_scores_val(fake_mol, mols, full_mols, full_smiles, vert, norm=True)     # 'mols' is output of Fake Reward
    #m1 =all_scores_chem(fake_mol, mols, vert, norm=True)
    #m0.update(m1)
    
    #maxlen = MolecularMetrics.max_component(mols, 45)
    
    #m0 = {k: np.array(v).mean() for k, v in m0.items()}
    #loss.update(m0)
    loss.update({'Valid': valid})
    loss.update({'Unique' unique})
    loss.update({'Novel': novel}) 
    #loss.update({'QED': statistics.mean(qed)})
    #loss.update({'SA': statistics.mean(sa)})
    #loss.update({'LogP': statistics.mean(logp)})
    #loss.update({'IntDiv': IntDiv})
    
    #wandb.log({"maxlen": maxlen})

    for tag, value in loss.items():
        
        log += ", {}: {:.4f}".format(tag, value)
    with open(log_path, "a") as f:
        f.write(log)  
        f.write("\n")                               
    print(log) 
    print("\n") 



def plot_attn(dataset_name, heads,attn_w, model, iter, epoch):
    
    cols = 4
    rows = int(heads/cols)

    fig, axes = plt.subplots( rows,cols, figsize = (30, 14))
    axes = axes.flat
    attentions_pos = attn_w[0]
    attentions_pos = attentions_pos.cpu().detach().numpy()
    for i,att in enumerate(attentions_pos):

        #im = axes[i].imshow(att, cmap='gray')
        sns.heatmap(att,vmin = 0, vmax = 1,ax = axes[i])
        axes[i].set_title(f'head - {i} ')
        axes[i].set_ylabel('layers')
    pltsavedir = "/home/atabey/attn/second"
    plt.savefig(os.path.join(pltsavedir, "attn" + model + "_" + dataset_name + "_"  + str(iter) + "_" + str(epoch) +  ".png"), dpi= 500,bbox_inches='tight')


def plot_grad_flow(named_parameters, model, iter, epoch):
    
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
            print(p.grad,n)
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
    pltsavedir = "/home/atabey/gradients/tryout"
    plt.savefig(os.path.join(pltsavedir, "weights_" + model  + "_"  + str(iter) + "_" + str(epoch) +  ".png"), dpi= 500,bbox_inches='tight')

"""
def _genDegree():
    
    ''' Generates the Degree distribution tensor for PNA, should be used everytime a different
        dataset is used.
        Can be called without arguments and saves the tensor for later use. If tensor was created
        before, it just loads the degree tensor.
        '''
    
    degree_path = os.path.join(self.degree_dir, self.dataset_name + '-degree.pt')
    if not os.path.exists(degree_path):
        
        
        max_degree = -1
        for data in self.dataset:
            d = geoutils.degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in self.dataset:
            d = geoutils.degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        torch.save(deg, 'DrugGEN/data/' + self.dataset_name + '-degree.pt')            
    else:    
        deg = torch.load(degree_path, map_location=lambda storage, loc: storage)
        
    return deg        
"""    
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
def fraction_unique(gen, k=None, n_jobs=1, check_validity=True):
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
        canonic = [i for i in canonic if i is not None]
        #raise ValueError("Invalid molecule passed to unique@k")
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

import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    
    # Solver for training and testing StarGAN.
    solver = Solver(config)

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--z_dim', type=int, default=3, help='dimension of domain labels')
    parser.add_argument('--g_conv_dim',default=[128,256,512], help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=[[128, 64], 128, [128, 64]], help='number of conv filters in the first layer of D')
    parser.add_argument('--lambda_gp', type=float, default=1, help='weight for gradient penalty')
    parser.add_argument('--post_method', type=str, default='soft_gumbel', choices=['softmax', 'soft_gumbel', 'hard_gumbel'])
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension.')
    parser.add_argument('--depth', type=int, default=3, help='Encoder depth.')
    parser.add_argument('--heads', type=int, default=8, help='Number of head for Attention.')
    parser.add_argument('--dec_depth', type=int, default=1, help='Decoder depth.')
    parser.add_argument('--dec_heads', type=int, default=8, help='Number of head for Decoder-Attention.')    
    parser.add_argument('--mlp_ratio', type=int, default=2, help='.')
    parser.add_argument('--drop_rate', type=float, default=0.1 , help='.')
    parser.add_argument('--warm_up_steps', type=float, default=0, help='.')
    parser.add_argument('--dis_select', type=str, default="PNA", help="conv, PNA, TraConv")
    parser.add_argument('--init_type', type=str, default="normal", help="u")
    parser.add_argument('--la', type=float, default=0.5, help="lambda value for RL and Generator loss balance")
    # PNA configurations
    parser.add_argument('--aggregators', type=str, default="max,mean,min,std", help='aggregator identifiers - "min","max","std","var","mean","sum"')
    parser.add_argument('--scalers', type=str, default="identity,attenuation,amplification", help='scaler identifiers - "attenuation","amplification","identity","linear", "inverse_linear')
    parser.add_argument('--pna_in_ch',type = int, default=50, help='PNA in channel dimension')
    parser.add_argument('--pna_out_ch', type=int, default=50, help='PNA out channel dimension')
    parser.add_argument('--edge_dim', type=int, default=50, help='PNA edge dimension')
    parser.add_argument('--towers', type=int, default=1, help='PNA towers')
    parser.add_argument('--pre_lay', type=int, default=1, help='Pre-transformation layer number')
    parser.add_argument('--post_lay', type=int, default=1, help='Post-transformation layer number')
    parser.add_argument('--pna_layer_num', type=int, default=1, help='PNA layers')
    parser.add_argument('--graph_add', type=str, default="set2set", help='global_add,set2set,graph_multitrans')   

    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=8, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.00001, help='learning rate for D')
    parser.add_argument('--g2_lr', type=float, default=0.0001, help='learning rate for G2')
    parser.add_argument('--d2_lr', type=float, default=0.00001, help='learning rate for D2')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--metrics', type=str, default="validity,unique", help='reward metrics')
    parser.add_argument('--feature_matching', type=str2bool, default=False, help='features for molecules')
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=10000, help='test model from this step')
    parser.add_argument('--num_test_epoch', type=int, default=1, help='inference epoch')
    parser.add_argument('--inference_sample_num', type=int, default=30000, help='inference samples')
    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    # Directories.
    parser.add_argument('--protein_data_dir', type=str, default='MolecularTransGAN-master/data/akt')      
    parser.add_argument('--drug_index', type=str, default='MolecularTransGAN-master/data/drug_smiles.index')  
    parser.add_argument('--drug_data_dir', type=str, default='MolecularTransGAN-master/data')    
    parser.add_argument('--mol_data_dir', type=str, default='MolecularTransGAN-master/data')
    parser.add_argument('--log_dir', type=str, default='MolecularTransGAN-master/molgan/logs')
    parser.add_argument('--model_save_dir', type=str, default='MolecularTransGAN-master/molgan/models')
    parser.add_argument('--sample_dir', type=str, default='MolecularTransGAN-master/molgan/samples')
    parser.add_argument('--result_dir', type=str, default='MolecularTransGAN-master/molgan/results')
    parser.add_argument('--degree_dir', type=str, default='MolecularTransGAN-master/data')
    parser.add_argument('--dataset_file', type=str, default='chembl25.pt')    
    # Step size.
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=100000)

    config = parser.parse_args()
    print(config)
    main(config)

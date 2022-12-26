import torch
import torch_geometric.utils as geoutils
from utils import *

def load_data(data, drugs, batch_size, device, b_dim, m_dim, drugs_b_dim, drugs_m_dim,z_dim,vertexes):

    z = sample_z(batch_size, z_dim)                                                   # (batch,max_len)          

    z = torch.from_numpy(z).to(device).float().requires_grad_(True)                
    data = data.to(device)
    drugs = drugs.to(device)                                               
    z_e = sample_z_edge(batch_size,vertexes,z_dim)                                                   # (batch,max_len,max_len)    
    z_n = sample_z_node(batch_size,vertexes,z_dim)                                                   # (batch,max_len)          
    z_edge = torch.from_numpy(z_e).to(device).float().requires_grad_(True)                                      # Edge noise.(batch,max_len,max_len)
    z_node = torch.from_numpy(z_n).to(device).float().requires_grad_(True)                                      # Node noise.(batch,max_len)       
    a = geoutils.to_dense_adj(edge_index = data.edge_index,batch=data.batch,edge_attr=data.edge_attr, max_num_nodes=int(data.batch.shape[0]/batch_size)) 
    x = data.x.view(batch_size,int(data.batch.shape[0]/batch_size),-1)

    a_tensor = label2onehot(a, b_dim, device)
    #x_tensor = label2onehot(x, m_dim)
    x_tensor = x

    a_tensor = a_tensor #+ torch.randn([a_tensor.size(0), a_tensor.size(1), a_tensor.size(2),1], device=a_tensor.device) * noise_strength_0
    x_tensor = x_tensor #+ torch.randn([x_tensor.size(0), x_tensor.size(1),1], device=x_tensor.device) * noise_strength_1

    drugs_a = geoutils.to_dense_adj(edge_index = drugs.edge_index,batch=drugs.batch,edge_attr=drugs.edge_attr, max_num_nodes=int(drugs.batch.shape[0]/batch_size))

    drugs_x = drugs.x.view(batch_size,int(drugs.batch.shape[0]/batch_size),-1)

    drugs_a = drugs_a.to(device).long() 
    drugs_x = drugs_x.to(device)
    drugs_a_tensor = label2onehot(drugs_a, drugs_b_dim,device).float()
    drugs_x_tensor = drugs_x

    drugs_a_tensor = drugs_a_tensor #+ torch.randn([drugs_a_tensor.size(0), drugs_a_tensor.size(1), drugs_a_tensor.size(2),1], device=drugs_a_tensor.device) * noise_strength_2
    drugs_x_tensor = drugs_x_tensor #+ torch.randn([drugs_x_tensor.size(0), drugs_x_tensor.size(1),1], device=drugs_x_tensor.device) * noise_strength_3
    #prot_n = akt1_human_annot[None,:].to(device).float()        
    #prot_e = akt1_human_adj[None,None,:].view(1,546,546,1).to(device).float()

            

    a_tensor_vec = a_tensor.reshape(batch_size,-1)
    x_tensor_vec = x_tensor.reshape(batch_size,-1)               
    real_graphs = torch.concat((x_tensor_vec,a_tensor_vec),dim=-1)                      

    a_drug_vec = drugs_a_tensor.reshape(batch_size,-1)
    x_drug_vec = drugs_x_tensor.reshape(batch_size,-1)               
    drug_graphs = torch.concat((x_drug_vec,a_drug_vec),dim=-1)  
    
    return drug_graphs, real_graphs, a_tensor, x_tensor, drugs_a_tensor, drugs_x_tensor, z, z_edge, z_node
import torch
import torch_geometric.utils as geoutils
from utils import label2onehot

def generate_z_values(batch_size=32, z_dim=32, vertexes=32, b_dim=32, m_dim=32, device=None):
    z = torch.normal(mean=0, std=1, size=(batch_size, z_dim), device=device) # (batch,max_len)
    z_edge = torch.normal(mean=0, std=1, size=(batch_size, vertexes, vertexes, b_dim), device=device) # (batch,max_len,max_len)
    z_node = torch.normal(mean=0, std=1, size=(batch_size, vertexes, m_dim), device=device) # (batch,max_len)

    z = z.float().requires_grad_(True)
    z_edge = z_edge.float().requires_grad_(True)                                      # Edge noise.(batch,max_len,max_len)
    z_node = z_node.float().requires_grad_(True)                                      # Node noise.(batch,max_len)
    return z, z_edge, z_node


def load_molecules(data=None, b_dim=32, m_dim=32, device=None, batch_size=32):
    data = data.to(device)
    a = geoutils.to_dense_adj(
        edge_index = data.edge_index,
        batch=data.batch,
        edge_attr=data.edge_attr,
        max_num_nodes=int(data.batch.shape[0]/batch_size)
    )
    x_tensor = data.x.view(batch_size,int(data.batch.shape[0]/batch_size),-1)
    a_tensor = label2onehot(a, b_dim, device)

    a_tensor_vec = a_tensor.reshape(batch_size,-1)
    x_tensor_vec = x_tensor.reshape(batch_size,-1)
    real_graphs = torch.concat((x_tensor_vec,a_tensor_vec),dim=-1)

    return real_graphs, a_tensor, x_tensor
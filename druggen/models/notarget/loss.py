from ...utils import matrices2mol
import torch

def discriminator_loss(
    generator=None,
    discriminator=None,
    mol_graph=None,
    batch_size=None,
    device=None,
    grad_pen=None,
    lambda_gp=None,
    z_edge=None,
    z_node=None
):
    
    # Compute loss with real molecules.
    
    logits_real_disc = discriminator(mol_graph)              
                        
    prediction_real =  - torch.mean(logits_real_disc)

    # Compute loss with fake molecules.

    node, edge, node_sample, edge_sample  = generator(z_edge,  z_node)

    graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
    
    logits_fake_disc = discriminator(graph.detach())
        
    prediction_fake = torch.mean(logits_fake_disc)
    
    # Compute gradient loss.
    
    eps = torch.rand(mol_graph.size(0),1).to(device)
    x_int0 = (eps * mol_graph + (1. - eps) * graph).requires_grad_(True)

    grad0 = discriminator(x_int0)
    d_loss_gp = grad_pen(grad0, x_int0) 
    
    # Calculate total loss
    
    d_loss = prediction_fake + prediction_real +  d_loss_gp * lambda_gp
    
    return node, edge,d_loss

    
def generator_loss(
    generator=None,
    discriminator=None,
    adj=None,
    annot=None,
    batch_size=None,
    
):
    
    # Compute loss with fake molecules.
   
    node, edge, node_sample, edge_sample  = generator(adj,  annot)
   
    
    graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
   
 
    logits_fake_disc = discriminator(graph)
    
    prediction_fake = - torch.mean(logits_fake_disc)
    
    # Produce molecules.

    g_edges_hat_sample = torch.max(edge_sample, -1)[1] 
    g_nodes_hat_sample = torch.max(node_sample , -1)[1]   
                
    fake_mol = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=dataset_name) 
            for e_, n_ in zip(g_edges_hat_sample, g_nodes_hat_sample)]        
    g_loss = prediction_fake
    
    
    return g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge

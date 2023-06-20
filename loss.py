
import torch

def discriminator_loss(generator, discriminator, mol_graph, adj, annot, batch_size, device, grad_pen, lambda_gp,z_edge,z_node):
    
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

    
def generator_loss(generator, discriminator, v, adj, annot, batch_size, penalty, matrices2mol, fps_r,submodel, dataset_name):
    
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
    # Compute penalty loss.
    if submodel == "RL":
        reward = penalty(fake_mol, fps_r)
    
        # Reinforcement Loss
   
        rew_fake = v(graph)

        reward_loss =  torch.mean(rew_fake) ** 2 + reward    
    
        # Calculate total loss
    
        g_loss = prediction_fake + reward_loss * 1
    
    
    return g_loss, fake_mol, g_edges_hat_sample, g_nodes_hat_sample, node, edge

def discriminator2_loss(generator, discriminator, mol_graph, adj, annot, batch_size, device, grad_pen, lambda_gp,akt1_adj,akt1_annot):

    # Generate molecules.
    
    dr_edges, dr_nodes = generator(adj, 
                                annot, 
                                akt1_adj,
                                akt1_annot)
    
    
    dr_edges_hat = dr_edges.view(batch_size, -1)
    
    dr_nodes_hat = dr_nodes.view(batch_size, -1)

    dr_graph = torch.cat((dr_nodes_hat, dr_edges_hat), dim=-1)                    

    # Compute loss with fake molecules.
    
    dr_logits_fake = discriminator(dr_graph.detach())

    d2_loss_fake = torch.mean(dr_logits_fake)
    
    # Compute loss with real molecules.
    
    dr_logits_real2 = discriminator(mol_graph)

    d2_loss_real = - torch.mean(dr_logits_real2)
    
    # Compute gradient loss.  
    
    eps_dr = torch.rand(mol_graph.size(0),1).to(device)
    x_int0_dr = (eps_dr * mol_graph + (1. - eps_dr) * dr_graph).requires_grad_(True)


    grad0_dr = discriminator(x_int0_dr)
    d2_loss_gp = grad_pen(grad0_dr, x_int0_dr) 
    
    # Compute total loss.
    
    d2_loss = d2_loss_fake + d2_loss_real + d2_loss_gp * lambda_gp
    
    return d2_loss

def generator2_loss(generator, discriminator, v, adj, annot, batch_size, penalty, matrices2mol, fps_r,ak1_adj,akt1_annot, submodel, drugs_name):
    
    # Generate molecules.
    
    dr_edges_g, dr_nodes_g = generator(adj, 
                                annot, 
                                ak1_adj,
                                akt1_annot)
    
    dr_edges_hat_g = dr_edges_g.view(batch_size, -1)
    
    dr_nodes_hat_g = dr_nodes_g.view(batch_size, -1)

    dr_graph_g = torch.cat((dr_nodes_hat_g, dr_edges_hat_g), dim=-1)      
    
    # Compute loss with fake molecules.

    dr_g_edges_hat_sample, dr_g_nodes_hat_sample = torch.max(dr_edges_g, -1)[1], torch.max(dr_nodes_g, -1)[1]
         
    g_tra_logits_fake2 = discriminator(dr_graph_g)      
   
    g2_loss_fake = - torch.mean(g_tra_logits_fake2)                                                            

    # Reward
    fake_mol_g = [matrices2mol(n_.data.cpu().numpy(), e_.data.cpu().numpy(), strict=True, file_name=drugs_name) 
                for e_, n_ in zip(dr_g_edges_hat_sample, dr_g_nodes_hat_sample)]       
    g2_loss =  g2_loss_fake    
    if submodel == "RL":
        reward2 = penalty(fake_mol_g, fps_r)
    
        # Reinforcement Loss
        
        rew_fake2 = v(dr_graph_g)
        reward_loss2 =  torch.mean(rew_fake2) ** 2 + reward2
        
        # Calculate total loss
        
        g2_loss =  g2_loss_fake + reward_loss2 * 10    
    
    return g2_loss, fake_mol_g, dr_g_edges_hat_sample, dr_g_nodes_hat_sample#, reward2

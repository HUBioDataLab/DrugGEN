import torch


def discriminator_loss(generator, discriminator, mol_graph, batch_size, device, grad_pen, lambda_gp, z_edge, z_node):
    # Compute loss with real molecules.
    logits_real_disc = discriminator(mol_graph)
    prediction_real =  - torch.mean(logits_real_disc)

    # Compute loss with fake molecules.
    node, edge, node_sample, edge_sample = generator(z_edge, z_node)
    graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
    logits_fake_disc = discriminator(graph.detach())
    prediction_fake = torch.mean(logits_fake_disc)

    # Compute gradient loss.
    eps = torch.rand(mol_graph.size(0),1).to(device)
    x_int0 = (eps * mol_graph + (1. - eps) * graph).requires_grad_(True)
    grad0 = discriminator(x_int0)
    d_loss_gp = grad_pen(grad0, x_int0) 

    # Calculate total loss
    d_loss = prediction_fake + prediction_real + d_loss_gp * lambda_gp
    return node, edge, d_loss


def generator_loss(generator, discriminator, adj, annot, batch_size):
    # Compute loss with fake molecules.
    node, edge, node_sample, edge_sample = generator(adj, annot)

    graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)

    logits_fake_disc = discriminator(graph)
    prediction_fake = - torch.mean(logits_fake_disc)
    g_loss = prediction_fake

    return g_loss, node, edge, node_sample, edge_sample

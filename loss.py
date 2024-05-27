import torch


def discriminator_loss(generator, discriminator, drug_edge, drug_node, batch_size, device, grad_pen, lambda_gp, z_edge, z_node, submodel):
    # Compute loss with real molecules.
    if submodel == "DrugGEN":
        logits_real_disc = discriminator(drug_edge, drug_node)
    else:
        logits_real_disc = discriminator(drug_node)

    prediction_real =  - torch.mean(logits_real_disc)

    # Compute loss with fake molecules.
    node, edge, node_sample, edge_sample = generator(z_edge, z_node)
    if submodel == "DrugGEN":
        logits_fake_disc = discriminator(edge_sample, node_sample)
    else:
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
        logits_fake_disc = discriminator(graph.detach())

    prediction_fake = torch.mean(logits_fake_disc)

    # Compute gradient penalty.
    eps_edge = torch.rand(batch_size, 1, 1, 1, device=device)  # Shape adapted for broadcasting with edges and nodes
    eps_node = torch.rand(batch_size, 1, 1, device=device)  # Shape adapted for broadcasting with edges and nodes
    int_node = eps_node * drug_node + (1 - eps_node) * node_sample
    int_edge = eps_edge * drug_edge + (1 - eps_edge) * edge_sample
    int_node.requires_grad_(True)
    int_edge.requires_grad_(True)

    # Compute discriminator output for interpolated samples
    if submodel == "DrugGEN":
        logits_interpolated = discriminator(int_edge, int_node)
    else:
        graph = torch.cat((int_node.view(batch_size, -1), int_edge.view(batch_size, -1)), dim=-1)
        logits_interpolated = discriminator(graph)

    # Compute gradient penalty for nodes and edges
    grad_penalty = grad_pen(logits_interpolated, int_node)

    # Calculate total discriminator loss
    d_loss = prediction_fake + prediction_real + lambda_gp * grad_penalty

    return node, edge, d_loss


def generator_loss(generator, discriminator, adj, annot, batch_size, submodel):
    # Compute loss with fake molecules.
    node, edge, node_sample, edge_sample = generator(adj, annot)
    if submodel == "DrugGEN":
        logits_fake_disc = discriminator(edge_sample, node_sample)
    else:
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
        logits_fake_disc = discriminator(graph)

    prediction_fake = - torch.mean(logits_fake_disc)

    g_loss = prediction_fake

    return g_loss, node, edge, node_sample, edge_sample

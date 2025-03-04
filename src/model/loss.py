import torch


def gradient_penalty(discriminator, real_node, real_edge, fake_node, fake_edge, batch_size, device, submodel):
    """
    Calculate gradient penalty for WGAN-GP.
    
    Args:
        discriminator: The discriminator model
        real_node: Real node features
        real_edge: Real edge features
        fake_node: Generated node features
        fake_edge: Generated edge features
        batch_size: Batch size
        device: Device to compute on
        submodel: Type of model being used ("DrugGEN" or other)
        
    Returns:
        Gradient penalty term
    """
    # Generate random interpolation factors
    eps_edge = torch.rand(batch_size, 1, 1, 1, device=device)
    eps_node = torch.rand(batch_size, 1, 1, device=device)
    
    # Create interpolated samples
    int_node = (eps_node * real_node + (1 - eps_node) * fake_node).requires_grad_(True)
    int_edge = (eps_edge * real_edge + (1 - eps_edge) * fake_edge).requires_grad_(True)

    if submodel == "DrugGEN":
        logits_interpolated = discriminator(int_edge, int_node)
        
        # Calculate gradients for both node and edge inputs
        weight = torch.ones(logits_interpolated.size(), requires_grad=False).to(device)
        gradients = torch.autograd.grad(
            outputs=logits_interpolated,
            inputs=[int_node, int_edge],
            grad_outputs=weight,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )
        
        # Combine gradients from both inputs
        gradients_node = gradients[0].view(batch_size, -1)
        gradients_edge = gradients[1].view(batch_size, -1)
        gradients = torch.cat([gradients_node, gradients_edge], dim=1)
    else:
        graph = torch.cat((int_node.view(batch_size, -1), int_edge.view(batch_size, -1)), dim=-1)
        logits_interpolated = discriminator(graph)
        weight = torch.ones(logits_interpolated.size(), requires_grad=False).to(device)
        dydx = torch.autograd.grad(
            outputs=logits_interpolated,
            inputs=graph,
            grad_outputs=weight,
            retain_graph=True,
            create_graph=True,
            only_inputs=True
        )[0]
        gradients = dydx.view(dydx.size(0), -1)
    
    # Calculate gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


def discriminator_loss(generator, discriminator, drug_adj, drug_annot, mol_adj, mol_annot, batch_size, device, lambda_gp, submodel):
    # Compute loss for drugs
    if submodel == "DrugGEN":
        logits_real_disc = discriminator(drug_adj, drug_annot)
    else:
        graph_drug = torch.cat((drug_annot.view(batch_size, -1), drug_adj.view(batch_size, -1)), dim=-1)
        logits_real_disc = discriminator(graph_drug)

    # Use mean reduction for more stable training
    prediction_real = -torch.mean(logits_real_disc)

    # Compute loss for generated molecules
    node, edge, node_sample, edge_sample, _ = generator(mol_adj, mol_annot)

    if submodel == "DrugGEN":
        logits_fake_disc = discriminator(edge_sample.detach(), node_sample.detach())
    else:
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
        logits_fake_disc = discriminator(graph.detach())

    prediction_fake = torch.mean(logits_fake_disc)

    # Compute gradient penalty using the new function
    gp = gradient_penalty(discriminator, drug_annot, drug_adj, node_sample.detach(), edge_sample.detach(), batch_size, device, submodel)

    # Calculate total discriminator loss
    d_loss = prediction_fake + prediction_real + lambda_gp * gp

    return node, edge, d_loss


def generator_loss(generator, discriminator, mol_adj, mol_annot, batch_size, submodel):
    # Compute loss with fake molecules
    node, edge, node_sample, edge_sample, _ = generator(mol_adj, mol_annot)

    if submodel == "DrugGEN":
        logits_fake_disc = discriminator(edge_sample, node_sample)
    else:
        graph = torch.cat((node_sample.view(batch_size, -1), edge_sample.view(batch_size, -1)), dim=-1)
        logits_fake_disc = discriminator(graph)
    
    prediction_fake = -torch.mean(logits_fake_disc)
    g_loss = prediction_fake

    return g_loss, node, edge, node_sample, edge_sample
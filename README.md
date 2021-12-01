# MolecularTransGAN
## Official implementation of MolecularTransGAN



There are several problems on implementation:

 - GAN-1 does not learn.
  
a. There may be a problem with Gumbel-Softmax. It does not produce binary elements for adjacency and annotation matrices. (From prior)
  
b. Softmax does not work properly neither. There may be a problem with combining annotation and adjacency matrices in GCN.

 - Graph agrregation class have dimension problem. 
 
a. Dimensions for sigmoid and tanh layer are hard-coded and should be changed.


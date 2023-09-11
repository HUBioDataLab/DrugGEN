
class CrossLossConfig:
    def __init__(
        self,
        act='relu',
        z_dim=16,
        max_atom=45,
        lambda_gp=1,
        dim=128,
        depth=1,
        heads=8,
        dec_depth=1,
        dec_heads=8,
        dec_dim=128,
        mlp_ratio=3,
        warm_up_steps=0,
        dis_select='mlp',
        init_type='normal',
        dropout=0.,
        dec_dropout=0.,
        n_critic=1,
        edges=None, # b_dim
        nodes=None, # m_dim
        clipping_value=2,
        add_features=False,
        # log_sample_step=1000,
        # resume=False,
        # resume_epoch=None,
        # resume_iter=None,
        # resume_directory=None,
        vertexes=None
    ):
        self.act = act
        self.z_dim = z_dim
        # self.max_atom = max_atom
        self.lambda_gp = lambda_gp
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dec_depth = dec_depth
        self.dec_heads = dec_heads
        self.dec_dim = dec_dim
        self.mlp_ratio = mlp_ratio
        self.warm_up_steps = warm_up_steps
        self.dropout = dropout
        self.dec_dropout = dec_dropout
        self.n_critic = n_critic
        # self.resume_iters = resume_iters
        self.clipping_value = clipping_value
        self.add_features = add_features
        self.vertexes = vertexes
        self.edges = edges
        self.nodes = nodes
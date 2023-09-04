from druggen import NoTargetConfig, NoTargetTrainer, NoTargetTrainerConfig, DruggenDataset
from torch_geometric.loader import DataLoader


dataset = DruggenDataset(
    "notarget/data", # every drug folder must have a raw.smi file
    max_atom=45,
)

dataloader = DataLoader(
    dataset, 
    shuffle=True,
    batch_size=128,
    drop_last=True
)

model_config = NoTargetConfig(
    act="relu",
    vertexes=int(dataloader.dataset[0].x.shape[0]),
    b_dim=len(dataset.bond_decoder_m),
    m_dim=len(dataset.bond_decoder_m),
)

trainer_config = NoTargetTrainerConfig(
    trainer_folder="notarget/trainer",
)

trainer = NoTargetTrainer(
    model_config=model_config,
    trainer_config=trainer_config,
)

trainer.train(dataset)

trainer.save_model("output_notarget/")

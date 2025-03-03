#  DrugGEN Paper Results
## Generated Molecules
- SMILES notations of 10,000 de novo generated molecules from DrugGEN model can be downloaded from [here](generated_molecules/DrugGEN_generated_molecules.csv). (In addition to this, the SMILES notations of 10,000 de novo generated molecules from the DrugGEN-NoTarget model [here](generated_molecules/DrugGEN-NoTarget_generated_molecules.csv)).
- We run our deep learning-based drug/compound-target protein interaction prediction system ([DEEPScreen](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc03414e)) on generated molecules from DrugGEN model. DEEPScreen predicted 5,700 of them as active against AKT1, 130 of which received the highest confidence score ([SMILES notations of DEEPScreen predicted actives](generated_molecules/DrugGEN_DEEPScreen_actives.csv)).
- At the same time, we conducted a molecular docking analysis on the de novo molecules generated from DrugGEN and other target-based generation models, including RELATION, TRIOMHPHE-BOA, ResGen, as well as on real AKT1 inhibitors, using the crystal structure of [AKT1](https://www.rcsb.org/structure/4gv1). A total of 1,600 molecules exhibited sufficiently low binding free energies (< -8 kcal/mol) for the DrugGEN model. The corresponding molecules can be found [here](generated_molecules/DrugGEN_generated_molecules_docking_th8.csv).
- Parallel to this, we applied filtering to 10,000 de novo generated molecules from the DrugGEN model using Lipinski, Veber, and PAINS filters. After this operation, 4,127 of them successfully passed the filters, and their SMILES notations can be found [here](generated_molecules/DrugGEN_generated_molecules_physicofilter.csv).
- Finally, de novo molecules to effectively target the AKT1 protein are selected via expert curation from the dataset of molecules with binding free energies lower than -8 kcal/mol and predicted as active by DEEPScreen against the AKT1 protein ([SMILES notations of the expert selected de novo AKT1 inhibitor molecules](generated_molecules/Selected_denovo_AKT1_inhibitors.csv)).
## Docking
[Glide (Schrödinger Suite)](https://www.schrodinger.com/products/glide) was used to perform docking of AKT1 inhibitors, randomly sampled 10K ChEMBL molecules and [DrugGEN generated molecules](generated_molecules), using AKT1 crystal structure [(4GV1)](https://www.rcsb.org/structure/4GV1) as a reference protein. The top 1,000 docking scores for each set are available [here](docking). Also, the docking results of the crystal structure and selected de novo molecule (MOL_01_027820) were visualized using [PyMOL](https://www.schrodinger.com/products/pymol) and saved as [PDB files](docking). 
## Molecular Dynamics (MD)
The simulation analyses were conducted for AKT1-Capivasertib complex (crystal structure: [4GV1](https://www.rcsb.org/structure/4gv1)) and AKT1-MOL_02_027820 complex (consisting of the 4GV1 protein and a de novo generated molecule) using the Simulation Interactions Diagram module integrated into Maestro ([Desmond (Schrödinger Suite)](https://www.schrodinger.com/products/desmond)). MD files for the [AKT1-Capivasertib complex](https://drive.google.com/drive/u/0/folders/1jLBZ7mIjbXnAwe_oNkO4uhdz5N8rgmm2) and [AKT1-MOL_02_027820 complex](https://drive.google.com/drive/u/0/folders/1jJcKbgVYNm5lLkhLe5EZ9waWtOCW7X5x) have been shared on Google Drive. 


## Evaluation Script

This script takes three arguments:
- `gen_smiles`: A list of SMILES strings representing the de novo generated molecules. Molecules should be found under a column named "SMILES".
- `ref_smiles_1`: A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. ChEMBL molecules)
- `ref_smiles_2`: A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. selected inhibitors)

The script calculates the following metrics:
- Validity: The fraction of valid molecules in the generated set.
- Uniqueness: The fraction of unique molecules in the generated set.
- Novelty: The fraction of molecules in the generated set that are not present in the reference sets.
- Internal Diversity: The average Tanimoto similarity between all pairs of molecules in the generated set.
- QED: The average QED score of the molecules in the generated set.
- SA: The average SA score of the molecules in the generated set.
- FCD: The average FCD score of the molecules in the generated set against both reference sets.
- Fragment Similarity: The average fragment similarity score of the molecules in the generated set against both reference sets.
- Scaffold Similarity: The average scaffold similarity score of the molecules in the generated set against both reference sets.
- Lipinski: The fraction of molecules in the generated set that pass the Lipinski filter.
- Veber: The fraction of molecules in the generated set that pass the Veber filter.
- PAINS: The fraction of molecules in the generated set that pass the PAINS filter.


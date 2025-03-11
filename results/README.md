##  DrugGEN Results
- SMILES notations of 10,000 de novo generated molecules from DrugGEN model can be downloaded from [here](generated_molecules/DrugGEN_generated_molecules_AKT1.csv) for AKT1 and from [here](generated_molecules/DrugGEN_generated_molecules_CDK2.csv) for CDK2.
- We conducted a molecular docking analysis on the de novo molecules generated from DrugGEN and other target-based generation models, including RELATION, TRIOMHPHE-BOA, ResGen, TargetDiff and Pocket2Mol as well as on real AKT1 and CDK2 inhibitors, using the crystal structure of [AKT1](https://www.rcsb.org/structure/4gv1) and [CDK2](https://www.rcsb.org/structure/4kd1), respectively. The top-performing 10% of molecules from docking analyses were compared across target-based methods for evaluation. The SMILES notations of these molecules and their docking scores are available [here](docking).
- Finally, de novo molecules to effectively target the AKT1 protein are selected via expert curation from the dataset of molecules with binding free energies lower than -8 kcal/mol and predicted as active by DEEPScreen against the AKT1 protein ([SMILES notations of the expert selected de novo AKT1 inhibitor molecules](generated_molecules/Selected_denovo_AKT1_inhibitors.csv)).

## Evaluation Script

This script takes three arguments:
- `gen_smiles`: A list of SMILES strings representing the de novo generated molecules. Molecules should be found under a column named "SMILES".
- `ref_smiles_1`: A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. ChEMBL molecules)
- `ref_smiles_2`(optional): A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. selected inhibitors)
- `output`: The output file where the computed metrics will be saved.

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


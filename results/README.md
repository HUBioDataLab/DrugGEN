##  DrugGEN Results
- SMILES notations of 10,000 de novo generated molecules from DrugGEN model can be downloaded from [here](generated_molecules/DrugGEN_generated_molecules_AKT1.csv) for AKT1 and from [here](generated_molecules/DrugGEN_generated_molecules_CDK2.csv) for CDK2.
- We conducted a molecular docking analysis on the de novo molecules generated from DrugGEN and other target-based generation models, including RELATION, TRIOMHPHE-BOA, ResGen, TargetDiff and Pocket2Mol as well as on real AKT1 and CDK2 inhibitors, using the crystal structure of [AKT1](https://www.rcsb.org/structure/4gv1) and [CDK2](https://www.rcsb.org/structure/4kd1), respectively. The top-performing 10% of molecules from docking analyses were compared across target-based methods for evaluation. The SMILES notations of these molecules and their docking scores are available [here](docking).
- Finally, de novo molecules to effectively target the AKT1 protein are selected via expert curation from the dataset of molecules with binding free energies lower than -8 kcal/mol and predicted as active by DEEPScreen against the AKT1 protein ([SMILES notations of the expert selected de novo AKT1 inhibitor molecules](generated_molecules/Selected_denovo_AKT1_inhibitors.csv)).

## Evaluation Script

This script takes four arguments:
- `gen`: A list of SMILES strings representing the de novo generated molecules. Molecules should be found under a column named "SMILES".
- `ref1`: A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. ChEMBL molecules)
- `ref2`(optional): A list of SMILES strings representing the reference molecules for novelty calculation. (e.g. selected inhibitors)
- `output (optional, default: results)`: The output file where the computed metrics will be saved.

The following is a generic example of how to use the evaluation script:

```bash
python evaluate.py --gen "[SMILES FILE]" --ref1 "[TRAINING SET FILE]" --ref2 "[TEST SET FILE]" --output "[PERFORMANCE RESULTS FILE]"
```

To evaluate the AKT1 targeted generated molecules used in the paper, run:

```bash
python evaluate.py --gen "generated_molecules/DrugGEN_generated_molecules_AKT1.csv" --ref1 "../data/chembl_train.smi" --ref2 "../data/akt_train.smi" --output "results_akt1"
```

To evaluate the CDK2 targeted generated molecules used in the paper, run:

```bash
python evaluate.py --gen "generated_molecules/DrugGEN_generated_molecules_CDK2.csv" --ref1 "../data/chembl_train.smi" --ref2 "../data/cdk2_train.smi" --output "results_cdk2.csv"
```

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


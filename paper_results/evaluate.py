import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED, AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from fcd_torch import FCD  # You'll need to install fcd_torch
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add parent directory to path
from utils import (fraction_valid, fraction_unique, novelty, 
                  internal_diversity,obey_lipinski, obey_veber, load_pains_filters, 
                     is_pains, FragMetric, ScafMetric)
import torch

class MoleculeEvaluator:
    def __init__(self, gen_smiles, ref_smiles_1, ref_smiles_2=None, n_jobs=8):
        """
        Initialize evaluator with generated and reference SMILES
        ref_smiles_2 is optional
        """
        self.gen_smiles = gen_smiles
        self.ref_smiles_1 = ref_smiles_1
        self.ref_smiles_2 = ref_smiles_2
        self.n_jobs = n_jobs
        
        # Convert SMILES to RDKit molecules and filter out invalid ones
        self.gen_mols = [mol for s in gen_smiles if s and (mol := Chem.MolFromSmiles(s)) is not None]
        self.ref_mols_1 = [mol for s in ref_smiles_1 if s and (mol := Chem.MolFromSmiles(s)) is not None]
        self.ref_mols_2 = [mol for s in ref_smiles_2 if s and (mol := Chem.MolFromSmiles(s)) is not None] if ref_smiles_2 else None
        
        # Initialize metrics that need setup
        self.fcd = FCD(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.pains_catalog = load_pains_filters()
        self.frag_metric = FragMetric(n_jobs=1)
        self.scaf_metric = ScafMetric(n_jobs=1)

    def calculate_basic_metrics(self):
        """Calculate validity, uniqueness, novelty, and internal diversity"""
        # Generate Morgan fingerprints for internal diversity calculation
        fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in self.gen_mols if mol is not None])
        
        results = {
            'validity': fraction_valid(self.gen_smiles, n_jobs=self.n_jobs),
            'uniqueness': fraction_unique(self.gen_smiles, n_jobs=self.n_jobs),
            'novelty_ref1': novelty(self.gen_smiles, self.ref_smiles_1, n_jobs=self.n_jobs),
            'internal_diversity': internal_diversity(fps)  # Pass fingerprints instead of molecules
        }
        if self.ref_smiles_2:
            results['novelty_ref2'] = novelty(self.gen_smiles, self.ref_smiles_2, n_jobs=self.n_jobs)
        return results

    def calculate_property_metrics(self):
        """Calculate QED and SA scores"""
        qed_scores = [QED.qed(mol) for mol in self.gen_mols if mol is not None]
        sa_scores = [sascorer.calculateScore(mol) for mol in self.gen_mols if mol is not None]
        
        return {
            'qed_mean': np.mean(qed_scores),
            'qed_std': np.std(qed_scores),
            'sa_mean': np.mean(sa_scores),
            'sa_std': np.std(sa_scores)
        }

    def calculate_fcd(self):
        """Calculate FCD score against both reference sets"""
        # Filter out None values and convert mols back to SMILES
        gen_valid_smiles = [Chem.MolToSmiles(mol) for mol in self.gen_mols if mol is not None]
        ref1_valid_smiles = [Chem.MolToSmiles(mol) for mol in self.ref_mols_1 if mol is not None]
        
        results = {
            'fcd_ref1': self.fcd(gen_valid_smiles, ref1_valid_smiles)
        }
        
        if self.ref_mols_2:
            ref2_valid_smiles = [Chem.MolToSmiles(mol) for mol in self.ref_mols_2 if mol is not None]
            results['fcd_ref2'] = self.fcd(gen_valid_smiles, ref2_valid_smiles)
            
        return results

    def calculate_similarity_metrics(self):
        """Calculate fragment and scaffold similarity"""

        results = {
            'frag_sim_ref1': self.frag_metric(gen=self.gen_mols, ref=self.ref_mols_1),
            'scaf_sim_ref1': self.scaf_metric(gen=self.gen_mols, ref=self.ref_mols_1)
        }
        if self.ref_mols_2:
            results.update({
                'frag_sim_ref2': self.frag_metric(gen=self.gen_mols, ref=self.ref_mols_2),
                'scaf_sim_ref2': self.scaf_metric(gen=self.gen_mols, ref=self.ref_mols_2)
            })
        return results

    def calculate_drug_likeness(self):
        """Calculate Lipinski, Veber and PAINS filtering results"""
        lipinski_scores = [obey_lipinski(mol) for mol in self.gen_mols if mol is not None]
        veber_scores = [obey_veber(mol) for mol in self.gen_mols if mol is not None]
        pains_results = [not is_pains(mol, self.pains_catalog) for mol in self.gen_mols if mol is not None]
        
        return {
            'lipinski_mean': np.mean(lipinski_scores),
            'lipinski_std': np.std(lipinski_scores),
            'veber_mean': np.mean(veber_scores),
            'veber_std': np.std(veber_scores),
            'pains_pass_rate': np.mean(pains_results)
        }

    def evaluate_all(self):
        """Run all evaluations and combine results"""
        results = {}
        
        print("\nCalculating basic metrics...")
        basic_metrics = self.calculate_basic_metrics()
        print("Basic metrics:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in basic_metrics.items()})
        results.update(basic_metrics)
        
        print("\nCalculating property metrics...")
        property_metrics = self.calculate_property_metrics()
        print("Property metrics:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in property_metrics.items()})
        results.update(property_metrics)
        
        print("\nCalculating FCD scores...")
        fcd_metrics = self.calculate_fcd()
        print("FCD metrics:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in fcd_metrics.items()})
        results.update(fcd_metrics)
        
        print("\nCalculating similarity metrics...")
        similarity_metrics = self.calculate_similarity_metrics()
        print("Similarity metrics:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in similarity_metrics.items()})
        results.update(similarity_metrics)
        
        print("\nCalculating drug-likeness metrics...")
        drug_likeness = self.calculate_drug_likeness()
        print("Drug-likeness metrics:", {k: f"{v:.3f}" if isinstance(v, float) else v for k, v in drug_likeness.items()})
        results.update(drug_likeness)
        
        return results

def evaluate_molecules_from_files(gen_path, ref_path_1, ref_path_2=None, smiles_col='SMILES', output_prefix="results", n_jobs=8):
    """
    Main function to evaluate generated molecules against reference sets, reading from CSV files
    
    Args:
        gen_path (str): Path to CSV file containing generated SMILES
        ref_path_1 (str): Path to CSV file containing first reference set SMILES
        ref_path_2 (str, optional): Path to CSV file containing second reference set SMILES
        smiles_col (str): Name of column containing SMILES strings
        output_prefix (str): Prefix for output files
        n_jobs (int): Number of parallel jobs
    """
    # Read SMILES from CSV files
    try:
        gen_df = pd.read_csv(gen_path)
        ref_df_1 = pd.read_csv(ref_path_1)
        ref_df_2 = pd.read_csv(ref_path_2) if ref_path_2 else None
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find one of the input CSV files: {e}")
    except pd.errors.EmptyDataError:
        raise ValueError("One of the input CSV files is empty")
    
    # Check if SMILES column exists
    for df, name in [(gen_df, 'generated'), (ref_df_1, 'reference 1')] + ([(ref_df_2, 'reference 2')] if ref_df_2 is not None else []):
        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found in {name} dataset")
    
    # Extract SMILES lists
    gen_smiles = gen_df[smiles_col].dropna().tolist()
    ref_smiles_1 = ref_df_1[smiles_col].dropna().tolist()
    ref_smiles_2 = ref_df_2[smiles_col].dropna().tolist() if ref_df_2 is not None else None
    
    # Validate that we have SMILES to process
    if not gen_smiles:
        raise ValueError("No valid SMILES found in generated set")
    if not ref_smiles_1 and not ref_smiles_2:
        raise ValueError("No valid SMILES found in one or both reference sets")
    
    print(f"\nProcessing datasets:")
    print(f"Generated molecules: {len(gen_smiles)}")
    print(f"Reference set 1: {len(ref_smiles_1)}")
    if ref_smiles_2:
        print(f"Reference set 2: {len(ref_smiles_2)}")
    
    # Run evaluation
    evaluator = MoleculeEvaluator(gen_smiles, ref_smiles_1, ref_smiles_2, n_jobs=n_jobs)
    results = evaluator.evaluate_all()
    
    print("\nSaving results...")
    # Add dataset sizes to results
    results.update({
        'n_generated': len(gen_smiles),
        'n_reference_1': len(ref_smiles_1),
        'n_reference_2': len(ref_smiles_2) if ref_smiles_2 is not None else 0
    })
    
    # Save results
    # Format float values to 3 decimal places for JSON
    formatted_results = {k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()}
    with open(f"{output_prefix}.json", 'w') as f:
        json.dump(formatted_results, f, indent=4)
    
    # Create DataFrame with formatted results
    df = pd.DataFrame([formatted_results])
    df.to_csv(f"{output_prefix}.csv", index=False)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate generated molecules against reference sets')
    parser.add_argument('--gen', required=True, help='Path to CSV file with generated SMILES')
    parser.add_argument('--ref1', required=True, help='Path to CSV file with first reference set SMILES')
    parser.add_argument('--ref2', help='Path to CSV file with second reference set SMILES (optional)')
    parser.add_argument('--smiles-col', default='SMILES', help='Name of SMILES column in CSV files')
    parser.add_argument('--output', default='results', help='Prefix for output files')
    parser.add_argument('--n-jobs', type=int, default=8, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_molecules_from_files(
            args.gen, 
            args.ref1, 
            args.ref2, 
            smiles_col=args.smiles_col,
            output_prefix=args.output,
            n_jobs=args.n_jobs
        )
        print(f"Evaluation complete. Results saved to {args.output}.json and {args.output}.csv")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

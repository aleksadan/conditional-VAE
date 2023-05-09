import rdkit 
from rdkit.Chem import Descriptors 
import numpy as np
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
from typing import *

###############################################################################
 
def maccs(
        mols: List[rdkit.Chem.rdchem.Mol]
        ) -> np.ndarray:
    """Calculates the MACCS keys for a list of molecules
    
    Args:
        mols:   list of molecules  
    
    Returns:
        an array (number of molecules, 167) of MACCS keys
    

    """
    fps = [MACCSkeys.GenMACCSKeys(x) for x in mols]
    a = np.zeros((len(mols), 167), dtype=np.int16)
    for i in range(len(mols)):
        DataStructs.ConvertToNumpyArray(fps[i], a[i])
    return a

#-----------------------------------------------------------------------------#

def ecfp(
        mols: List[rdkit.Chem.rdchem.Mol], 
        radius: int = 2, 
        bits: int = 1024
        ) -> np.ndarray:
    """Calculates the ECFPs for a list of molecules
    
    Args:
        mols:   list of molecules  
        radius: radius for fragment calculation
        bits:  bits available for folding ECFP


    Returns:
        an array (number of molecules, 1024) of ECFPs


    """
    fps = [AllChem.GetMorganFingerprintAsBitVect(x,int(radius),nBits=int(bits)) for x in mols]
    a = np.zeros((len(mols), 1024), dtype=np.int16)
    for i in range(len(mols)):
        DataStructs.ConvertToNumpyArray(fps[i], a[i])
    return a

#-----------------------------------------------------------------------------#

def pc_props(
        mols: List[rdkit.Chem.rdchem.Mol]
             ) -> np.ndarray:
    """Calculates the molecular descriptors available in rdkit for a list of molecules
    
    Args:
        mols:   list of molecules  
    
    Returns:
        an array (number of molecules, 208) of molecular descriptors

    """
    calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
    names = calc.GetDescriptorNames()
    mol_desc =[]
    for mol in mols:
       descriptors = calc.CalcDescriptors(mol)
       mol_desc.append(descriptors)
    mol_desc = np.array(mol_desc)
    mol_desc = np.clip(mol_desc, -10000, 10000)
    mol_desc = np.nan_to_num(mol_desc)
    return  mol_desc 

  

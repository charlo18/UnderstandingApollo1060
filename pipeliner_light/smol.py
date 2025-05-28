'''
this part normalise the molecule to the smiles format and suppress some abnormal molecule
'''

import os

import numpy as np
#for molecular validation and standardisation
from molvs import Standardizer

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdchem import Mol

from .descriptors import feat_dict, rdkit_headers, fingerprint_headers

#gets the current path of file
current_path = os.path.dirname(os.path.abspath(__file__))
#create the standardizer
standardizer = Standardizer()
#create a strip version of a molecule from the file (current_path + salt.txt) which is located in files of this projet
salt_remover = SaltRemover(defnFilename=os.path.join(current_path, 'files', 'Salts.txt'))

#gets rid of the inifinite and the NaN
def mp_featurize(mol, feat_info):
    mol.featurize(feat_info)
    if not np.all(np.isfinite(mol.features_values)) or np.any(np.isnan(mol.features_values)):
        mol.mol_block = None


class SMol:
    #initial build of the SMol class
    def __init__(self, source, endpoints=None, id=None, standardization=1):
        if isinstance(source, str):
            source = Chem.MolFromSmiles(source)
        elif isinstance(source, Mol):
            source = source
        else:
            raise TypeError('Unsupported source type')

        if endpoints is None:
            self.endpoints = source.GetPropsAsDict()
        else:
            self.endpoints = endpoints
        if standardization == 1:
            rdk_mol, self.stand_changes = self.standardize(source)

            self.mol_block = Chem.MolToMolBlock(rdk_mol) if rdk_mol is not None else None
        else:
            self.mol_block = Chem.MolToMolBlock(source) if source is not None else None
        self.id = id
        self._scaffold = None
        self._smiles = None
        self.features_names = None
        self.features_values = None

    #defining property for smiles and rmol
    @property
    def smiles(self):
        if self.mol_block is not None:
            if self._smiles is None:
                self._smiles = Chem.MolToSmiles(Chem.MolFromMolBlock(self.mol_block))
            return self._smiles
        else:
            return None

    @property
    def rmol(self):
        if self.mol_block is not None:
            return Chem.MolFromMolBlock(self.mol_block)
        else:
            return None

    #for each type (Descs,ECFP) adds the name and the values
    def featurize(self, features_info):

        features_names = []
        features_values = []
        for feature_info in features_info:
            if feature_info['type'] == 'DESCS':
                features_names.extend(rdkit_headers())
            elif feature_info['type'] == 'ECFP':
                if feature_info.get('length') == None:
                    feature_info['length'] = 1024
                if feature_info.get('radius') == None:
                    feature_info['radius'] = 2
                features_names.extend(fingerprint_headers(feature_info))

            features_values.extend(feat_dict[feature_info['type']](self.rmol, feature_info))

        self.features_names = np.array(features_names)
        self.features_values = np.array(features_values)

    #create a dictionnary data
    @property
    def features(self):
        return {feature_name: features_value for feature_name, features_value in
                zip(self.features_names, self.features_values)}

    #transform a molecule into a smiles
    @staticmethod
    def standardize(rdkit_mol, mode=1):

        if mode == 1:

            in_smiles = Chem.MolToSmiles(rdkit_mol)
            smol = standardizer.isotope_parent(rdkit_mol)
            smol = salt_remover.StripMol(smol)
            st_smiles = Chem.MolToSmiles(smol)
            if st_smiles in ('', '.'):
                return None, None
            if '.' in st_smiles:
                components = st_smiles.split('.')
                num_unique_components = len(set(components))
                if num_unique_components != 1:
                    return None, None
                else:
                    smol = Chem.MolFromSmiles(components[0])

            smol = standardizer.standardize(smol)
            st_smiles = Chem.MolToSmiles(smol)
            smol = Chem.MolFromSmiles(st_smiles)
            st_smiles = Chem.MolToSmiles(smol)
            if in_smiles != st_smiles:
                changes = {in_smiles: st_smiles}
            else:
                changes = None
            return smol, changes

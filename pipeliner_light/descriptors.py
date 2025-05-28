"""
this file implement 2 main routines:
one for ecfp, and one for rdkit_descriptors
They both have the same goal of encoding descriptors automatically
featurize_molecule create a dictionnary that depends on the type of the feature.
This file also implements headers for the descriptors display"""
import os

from rdkit.Chem import Descriptors, MolToSmiles, AllChem

#get the directory path to open the descriptors_list file
current_path = os.path.dirname(os.path.abspath(__file__))
descriptors_list = []
with open(os.path.join(current_path, 'files', 'descriptors_list.txt'), "r") as f:
    for line in f:
        descriptors_list.append(line.strip())

#create a dictionnary with the descriptors list
descriptors_dict = dict(Descriptors.descList)

#decide what function to call based on the feature type
def featurize_molecule(mol, features):
    features_list = []
    for feature in features:
        #feat_dict[feature['type']](mol, feature) -> call either ecfp or rdkit_descriptor with (mol,feature) as an argument
        features_list.extend(feat_dict[feature['type']](mol, feature))
    return features_list

#Returns a Morgan fingerprint for a molecule as a bit vector with a certain length and radius
def ecfp(molecule, options):
    return [x for x in AllChem.GetMorganFingerprintAsBitVect(
        molecule, options['radius'], options['length'])]

#-------------------------------------------------------------
#this part is to build a header for display
#get the first element(name) of each rows of the descriptors list
def rdkit_headers():
    headers = [x[0] for x in Descriptors.descList]
    return headers

#build columns names
def fingerprint_headers(options):
    return ['{}{}_{}'.format(options['type'], options['radius'], x) for x in range(options['length'])]
#--------------------------------------------------------------

#go through all the descriptors list
def rdkit_descriptors(molecule, options=None):
    descriptors = []
    for desc_name in descriptors_list:
        try:
            #verify the value of each molecule
            desc = descriptors_dict[desc_name]
            bin_value = desc(molecule)

            #if error raise exception
        except (ValueError, TypeError, ZeroDivisionError) as exception:
            print(
                'Calculation of the Descriptor {} failed for a molecule {} due to {}'.format(
                    str(desc_name), str(MolToSmiles(molecule)), str(exception))
            )
            bin_value = 'NaN'
        #makes a new list with NaN instead of error as values
        descriptors.append(bin_value)

    return descriptors

#mapping descriptors to there function
feat_dict = {"ECFP": ecfp, "DESCS": rdkit_descriptors}

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Define file paths
input_file_path = 'input_data.csv'  # Replace with your input file path
output_file_path = 'output_data_with_descriptors.csv'  # Replace with your output file path

# Read input data
data = pd.read_csv(input_file_path)

# Get all available descriptor names from RDKit
descriptor_names = [desc[0] for desc in Descriptors.descList]

# Define a function to calculate all descriptors for a given molecule
def calculate_all_descriptors(mol):
    descriptors = {}
    for desc_name in descriptor_names:
        try:
            # Retrieve descriptor function dynamically and calculate its value
            desc_func = getattr(Descriptors, desc_name)
            descriptors[desc_name] = desc_func(mol)
        except:
            # If a descriptor cannot be calculated, set it to None
            descriptors[desc_name] = None
    return descriptors

# Initialize lists to store descriptors for each SMILES column
descriptors_smiles1 = []
descriptors_smiles2 = []

# Calculate descriptors for each row's SMILES1 and SMILES2
for index, row in data.iterrows():
    mol1 = Chem.MolFromSmiles(row['SMILES1'])
    mol2 = Chem.MolFromSmiles(row['SMILES2'])

    if mol1:
        desc1 = calculate_all_descriptors(mol1)
        descriptors_smiles1.append(desc1)
    else:
        # If SMILES is invalid, add empty values
        descriptors_smiles1.append({name: None for name in descriptor_names})

    if mol2:
        desc2 = calculate_all_descriptors(mol2)
        descriptors_smiles2.append(desc2)
    else:
        descriptors_smiles2.append({name: None for name in descriptor_names})

# Convert descriptor lists to DataFrames and add suffixes to distinguish them
descriptors_df1 = pd.DataFrame(descriptors_smiles1).add_suffix('_1')
descriptors_df2 = pd.DataFrame(descriptors_smiles2).add_suffix('_2')

# Concatenate descriptors with the original data
data_with_descriptors = pd.concat([data, descriptors_df1, descriptors_df2], axis=1)

# Save results to a new file
data_with_descriptors.to_csv(output_file_path, index=False)

print("Descriptors have been calculated and saved to the output file.")

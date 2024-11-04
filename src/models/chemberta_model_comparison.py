import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyRegressor

# Configure display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load ChemBERTa model and tokenizer
chemberta_model = RobertaModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
chemberta_model.to(device)
chemberta_model.eval()
tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

# Define file paths
input_file_path = 'input_data_with_descriptors.csv'  # Replace with your input file path

# Load data
data = pd.read_csv(input_file_path)

# Select features and target variable
X_features = data[['X#1 (molar fraction)', 'X#2 (molar fraction)', 'T#1', 'T#2']]
y = data['DES melting temperature, K']

# Define function to extract ChemBERTa features for a SMILES string
def get_chemberta_features(smiles):
    inputs = tokenizer(smiles, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = chemberta_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Extract ChemBERTa features for SMILES1 and SMILES2
smiles1_features = np.vstack([get_chemberta_features(smiles) for smiles in data['SMILES1']])
smiles2_features = np.vstack([get_chemberta_features(smiles) for smiles in data['SMILES2']])

# Concatenate features from SMILES1 and SMILES2
smiles_features = np.hstack([smiles1_features, smiles2_features])

# Combine numerical features with ChemBERTa features
X = np.hstack([X_features.values, smiles_features])

# Use SimpleImputer to fill missing values with the column mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use LazyRegressor to evaluate multiple regression models
multiple_ml_model = LazyRegressor(verbose=0, ignore_warnings=True, predictions=True)
models, predictions = multiple_ml_model.fit(X_train, X_test, y_train, y_test)

# Print model comparison results
print("Model comparison results:")
print(models)
print("\nPrediction results:")
print(predictions)

models.to_csv("output_model_results.csv", index=False)
predictions.to_csv("output_predictions.csv", index=False)

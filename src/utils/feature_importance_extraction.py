import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

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
output_file_path = 'feature_importances.csv'  # Replace with your output file path

# Load data
data = pd.read_csv(input_file_path)

# Select numerical features, excluding SMILES1, SMILES2, and target columns
X_numerical = data.drop(columns=['SMILES1', 'SMILES2', 'DES melting temperature, K'])
y = data['DES melting temperature, K']

# Define function to extract ChemBERTa features for SMILES strings
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
X = np.hstack([X_numerical.values, smiles_features])

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)

# Extract feature importances
feature_importances = model.feature_importances_

# Calculate total importance for SMILES1 and SMILES2 features
num_numerical_features = X_numerical.shape[1]
num_chemberta_features = smiles1_features.shape[1]

# Importance for numerical features
numerical_importance = feature_importances[:num_numerical_features]

# Total importance for SMILES1 and SMILES2 ChemBERTa features
smiles1_importance = feature_importances[num_numerical_features:num_numerical_features + num_chemberta_features].sum()
smiles2_importance = feature_importances[num_numerical_features + num_chemberta_features:].sum()

# Create a DataFrame to display feature importances
feature_names = list(X_numerical.columns) + ["SMILES1_ChemBERTa", "SMILES2_ChemBERTa"]
importance_values = list(numerical_importance) + [smiles1_importance, smiles2_importance]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_values
})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display feature importances
print("Feature importance results:")
print(importance_df)

# Save feature importances to a CSV file
importance_df.to_csv(output_file_path, index=False)

# Output file path
print(f"Feature importance results have been saved to: {output_file_path}")

import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from transformers import RobertaModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Set global font to Times New Roman
rc('font', family='Times New Roman')

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

# Define file path
input_file_path = 'input_data_with_descriptors.csv'  # Replace with your input file path

# Load data
data = pd.read_csv(input_file_path)

# Select specific numerical features
selected_columns = [
    'X#2 (molar fraction)', 'X#1 (molar fraction)', 'T#2', 'T#1', 'NumHAcceptors_2',
    'Chi0_2', 'PEOE_VSA9_1', 'PEOE_VSA11_2', 'Kappa1_2', 'HeavyAtomCount_2',
    'MolWt_2', 'HeavyAtomMolWt_2', 'SPS_2', 'VSA_EState3_2', 'MolLogP_2',
    'BCUT2D_LOGPHI_2', 'PEOE_VSA11_1', 'BCUT2D_CHGLO_2', 'BCUT2D_MWLOW_1',
    'NumValenceElectrons_2', 'Kappa3_2', 'BCUT2D_LOGPHI_1'
]
X_numerical = data[selected_columns]
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

# Calculate the sum of features for SMILES1 and SMILES2 and merge them into a single feature
smiles1_sum_features = smiles1_features.sum(axis=1)
smiles2_sum_features = smiles2_features.sum(axis=1)
chemberta_smiles_sum = smiles1_sum_features + smiles2_sum_features

# Add the combined feature to the dataset and drop other ChemBERTa features
X = X_numerical.copy()  # Use numerical features
X['ChemBERTa_SMILES_Sum'] = chemberta_smiles_sum  # Add combined feature

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Calculate correlation matrix
corr_matrix = X.corr()

# Plot heatmap for the correlation matrix
plt.figure(figsize=(12, 10))  # Set figure size

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "ticks": [-1, -0.5, 0, 0.5, 1]},  # Set color bar range
    annot_kws={"size": 8, "weight": "bold", "color": "black"}  # Set annotation font style
)

plt.xticks(rotation=45, ha='right', fontsize=10, weight='bold')  # Adjust x-axis labels
plt.yticks(rotation=0, fontsize=10, weight='bold')  # Adjust y-axis labels
plt.show()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and predict using ExtraTreesRegressor
model = ExtraTreesRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)

# Print model evaluation results
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("ExtraTreesRegressor model evaluation results:")
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)
print("\nPrediction results:")
print(y_pred)

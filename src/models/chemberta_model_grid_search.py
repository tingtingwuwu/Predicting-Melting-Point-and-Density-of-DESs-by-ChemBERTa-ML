import pandas as pd
import torch
import numpy as np
from transformers import RobertaModel, AutoTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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

# Select features and target variable
X_features = data[['X#1 (molar fraction)', 'X#2 (molar fraction)', 'T#1', 'T#2']]
y = data['DES melting temperature, K']

# Function to extract ChemBERTa features for SMILES strings
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

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models with parameter grids
models = {
    "XGBRegressor": {
        "model": XGBRegressor(objective='reg:squarederror'),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 6, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        }
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "LGBMRegressor": {
        "model": LGBMRegressor(),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'num_leaves': [31, 50, 70, 100],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'boosting_type': ['gbdt', 'dart'],
            'subsample': [0.6, 0.8, 1.0]
        }
    },
    "HistGradientBoostingRegressor": {
        "model": HistGradientBoostingRegressor(),
        "param_grid": {
            'max_iter': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_leaf': [20, 30, 40]
        }
    },
    "ExtraTreesRegressor": {
        "model": ExtraTreesRegressor(),
        "param_grid": {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
}

# Train and evaluate models with grid search
results = []
for name, config in models.items():
    model = config["model"]
    param_grid = config["param_grid"]
    # Grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=10, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    aard = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Average Absolute Relative Deviation in %
    results.append({
        "Model": name,
        "Best Parameters": grid_search.best_params_,
        "Mean Squared Error (MSE)": mse,
        "R^2 Score": r2,
        "Average Absolute Relative Deviation (AARD) %": aard
    })
    print(f"\n{name} model evaluation results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R^2 Score: {r2}")
    print(f"Average Absolute Relative Deviation (AARD): {aard}%")

# Display results for all models
df_results = pd.DataFrame(results)
print("\nEvaluation results for all models:")
print(df_results)

# Predicting Properties of Deep Eutectic Solvents (DESs) using ChemBERTa and Ensemble Models

## Abstract

Deep eutectic solvents (DESs) have emerged as sustainable alternatives to traditional solvents due to their low toxicity, biodegradability, and versatility. However, accurately predicting the melting point and density of DESs is challenging due to their structural diversity and complexity. Conventional methods often struggle to capture the nuanced structural details encoded in SMILES representations, limiting predictive precision. 

To address this, our study utilizes ChemBERTa, a pre-trained Transformer model, to extract high-dimensional embeddings from SMILES strings, effectively capturing complex molecular interactions and subtle structural features. Through feature importance analysis, we identified missing molecular information in the ChemBERTa embeddings and supplemented it with select physicochemical descriptors from RDKit, creating a feature set that enhances both interpretability and predictive accuracy.

Optimized ensemble models, including ExtraTreesRegressor (ETR) and XGBRegressor (XGBR), are then applied to this enriched feature set, achieving notable improvements in prediction accuracy for DESs melting point and density. Rigorous grid search and ten-fold cross-validation ensure model robustness and generalizability. Experimental results confirm the effectiveness of this approach, underscoring the transformative role of pre-trained deep learning models in chemical informatics and supporting scalable, sustainable DESs design.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data](#data)
- [Code Modules](#code-modules)
- [Usage](#usage)
- [Results](#results)
- [System Requirements and Configuration](#system-requirements-and-configuration)

---

## Project Structure

The directory structure of this project is organized as follows:
Model architecture/ ├── README.md # Project description and usage instructions ├── data/ # Data folder │ ├── init.py │ ├── descriptor_extraction.py # Script for extracting descriptors from molecules │ ├── DESs_density_data.csv # Density data for DESs │ └── DESs_melting_point_data.csv # Melting point data for DESs ├── output/ # Output folder (for generated files) ├── src/ # Source code │ ├── init.py │ ├── models/ # Model scripts │ │ ├── init.py │ │ ├── chemberta_model_comparison.py │ │ └── chemberta_model_grid_search.py │ └── utils/ # Utility scripts │ ├── init.py │ ├── correlation_heatmap.py │ └── feature_importance_extraction.py └── requirements.txt # Project dependencies


## Dependencies

Install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```
## Key Libraries
pandas: For data manipulation
torch and transformers: For using the ChemBERTa model
scikit-learn: For machine learning models, preprocessing, and evaluation
seaborn and matplotlib: For data visualization
numpy: For numerical computations

## Code Modules
### Data Preparation and Descriptor Extraction
descriptor_extraction.py: Extracts molecular descriptors from SMILES strings using RDKit and ChemBERTa embeddings. These descriptors are saved and used as input features for model training.
### Models
chemberta_model_comparison.py: Trains and compares various machine learning models (e.g., ExtraTreesRegressor, RandomForestRegressor) on the prepared dataset to predict DES properties.
chemberta_model_grid_search.py: Performs grid search and cross-validation to optimize hyperparameters for selected models, improving predictive accuracy and robustness.
### Utilities
correlation_heatmap.py: Generates a heatmap to visualize the correlation matrix for selected features, helping to identify relationships and redundancy among descriptors.
feature_importance_extraction.py: Analyzes feature importance from trained models, highlighting key descriptors that influence DES property predictions.
## Usage
### Set up the Environment
Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```
### Step-by-Step Execution
#### Step 2.1: Data Preparation
Run descriptor_extraction.py to extract molecular descriptors and ChemBERTa embeddings. Ensure the input data (e.g., DESs_density_data.csv, DESs_melting_point_data.csv) is in the data/ directory.

```bash
python data/descriptor_extraction.py
```
#### Step 2.2: Model Training and Comparison
Execute chemberta_model_comparison.py to train and evaluate different models on the extracted features.

```bash
python src/models/chemberta_model_comparison.py
```
#### Step 2.3: Hyperparameter Tuning
Run chemberta_model_grid_search.py to perform grid search with cross-validation on selected models, improving predictive performance.

```bash
python src/models/chemberta_model_grid_search.py
```
#### Step 2.4: Generate Correlation Heatmap
Use correlation_heatmap.py to visualize the feature correlation matrix.

```bash
python src/utils/correlation_heatmap.py
```
#### Step 2.5: Analyze Feature Importance
Run feature_importance_extraction.py to calculate and display feature importances for each model.

```bash
python src/utils/feature_importance_extraction.py
```
## Results
After following the steps above, results such as model evaluation metrics, feature importances, and correlation heatmaps will be available. You may also save these results in the output/ directory for further analysis.

### Example Outputs
#### Model Evaluation:
Metrics such as Mean Squared Error (MSE), R-squared score (R²), and Average Absolute Relative Deviation (AARD) for each model.
#### Feature Importance:
A ranked list of features by their importance, allowing insight into which molecular descriptors most influence DES property predictions.
#### Correlation Heatmap:
A visual representation of feature correlations, highlighting redundant features and providing a deeper understanding of the dataset.

## System Requirements and Configuration
### Recommended System Specifications
The project requires a modern machine with the following specifications to run efficiently, especially for model training using ChemBERTa embeddings:

  Operating System: Windows 10 (version 19045)
  Processor: Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz, with 14 physical cores and 20 logical processors, and a maximum frequency of 3.5GHz.
  Graphics Card: NVIDIA GeForce RTX 4060 Ti with 16GB of dedicated VRAM.
  Memory: At least 16GB of RAM (more is recommended for handling large datasets and model training).
Note: The project leverages GPU acceleration for ChemBERTa feature extraction. Ensure that torch is configured to utilize the GPU (CUDA) for faster computation if you have an NVIDIA GPU.

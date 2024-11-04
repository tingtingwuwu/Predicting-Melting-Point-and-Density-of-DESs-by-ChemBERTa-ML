```markdown
# Predicting Properties of Deep Eutectic Solvents (DESs) using ChemBERTa and Ensemble Models

## Abstract

Deep eutectic solvents (DESs) are gaining attention as sustainable solvent alternatives due to their low toxicity, biodegradability, and broad applicability in various fields. However, predicting critical properties like melting point and density is complex due to DESs' structural heterogeneity. Conventional prediction models often fail to accurately capture subtle structural intricacies, particularly those encoded within SMILES (Simplified Molecular Input Line Entry System) representations. 

Our study employs ChemBERTa, a Transformer-based deep learning model, to extract detailed embeddings from SMILES strings, effectively capturing molecular interactions and nuanced structural features. Recognizing limitations in ChemBERTa embeddings, we enhance these with specific physicochemical descriptors from RDKit, creating a comprehensive feature set that balances predictive power with interpretability.

Ensemble models, including ExtraTreesRegressor (ETR) and XGBRegressor (XGBR), are trained on this enriched feature set, yielding significant improvements in prediction accuracy for DES melting point and density. Through extensive grid search and ten-fold cross-validation, we ensure model robustness and generalizability. Our results highlight the potential of combining pre-trained deep learning models with ensemble techniques to drive scalable and sustainable DES design.

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dependencies](#dependencies)
3. [Data](#data)
4. [Code Modules](#code-modules)
5. [Usage](#usage)
6. [Results](#results)
7. [System Requirements and Configuration](#system-requirements-and-configuration)

---

## Project Structure

The project is structured as follows:

```
Model architecture/
├── README.md                 # Project description and usage instructions
├── data/                     # Data folder
│   ├── __init__.py
│   ├── descriptor_extraction.py  # Extracts molecular descriptors from SMILES
│   ├── DESs_density_data.csv     # Density data for DESs
│   └── DESs_melting_point_data.csv  # Melting point data for DESs
├── output/                   # Output folder for generated files
├── src/                      # Source code
│   ├── __init__.py
│   ├── models/               # Model scripts
│   │   ├── __init__.py
│   │   ├── chemberta_model_comparison.py
│   │   └── chemberta_model_grid_search.py
│   └── utils/                # Utility scripts
│       ├── __init__.py
│       ├── correlation_heatmap.py
│       └── feature_importance_extraction.py
└── requirements.txt          # Project dependencies
```

## Dependencies

To ensure compatibility, install all necessary Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Libraries

- **pandas**: For data manipulation and preprocessing.
- **torch** and **transformers**: For loading and utilizing ChemBERTa for molecular embedding extraction.
- **scikit-learn**: For building, training, and evaluating machine learning models.
- **seaborn** and **matplotlib**: For data visualization.
- **numpy**: For handling numerical computations efficiently.

## Data

The project utilizes two main datasets:

- **DESs_density_data.csv**: Contains density measurements for various DES compounds.
- **DESs_melting_point_data.csv**: Contains melting point data for the same DES compounds.

Both datasets are stored in the `data/` directory and are used as input for descriptor extraction and feature generation.

## Code Modules

### Data Preparation and Descriptor Extraction

- **descriptor_extraction.py**: This script generates two types of features:
  - **ChemBERTa embeddings** from SMILES strings, capturing intricate molecular patterns and relationships.
  - **RDKit descriptors** to augment ChemBERTa features with specific physicochemical properties for better model interpretability.
  
  The extracted features are saved for model training.

### Models

- **chemberta_model_comparison.py**: Trains multiple models (e.g., ExtraTreesRegressor, RandomForestRegressor) to predict DES properties using the prepared dataset and compares performance metrics.
- **chemberta_model_grid_search.py**: Conducts grid search and cross-validation to optimize hyperparameters, boosting model accuracy and robustness.

### Utilities

- **correlation_heatmap.py**: Generates a heatmap to visualize feature correlations, aiding in identifying redundant or highly correlated descriptors.
- **feature_importance_extraction.py**: Extracts and ranks features based on their importance, providing insights into which descriptors most significantly influence predictions.

## Usage

### Set up the Environment

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

#### Step 1: Data Preparation

Run `descriptor_extraction.py` to extract molecular descriptors and ChemBERTa embeddings. Ensure the input data files (`DESs_density_data.csv`, `DESs_melting_point_data.csv`) are placed in the `data/` directory.

```bash
python data/descriptor_extraction.py
```

#### Step 2: Model Training and Comparison

Execute `chemberta_model_comparison.py` to train and evaluate different models on the extracted features, comparing performance across models to identify the best fit.

```bash
python src/models/chemberta_model_comparison.py
```

#### Step 3: Hyperparameter Tuning

Run `chemberta_model_grid_search.py` to perform grid search and cross-validation on selected models, further enhancing predictive performance.

```bash
python src/models/chemberta_model_grid_search.py
```

#### Step 4: Generate Correlation Heatmap

Use `correlation_heatmap.py` to visualize the correlation matrix for all features, aiding in understanding feature relationships and identifying possible redundancies.

```bash
python src/utils/correlation_heatmap.py
```

#### Step 5: Analyze Feature Importance

Run `feature_importance_extraction.py` to compute and display the importance of each feature, highlighting descriptors that significantly impact DES property predictions.

```bash
python src/utils/feature_importance_extraction.py
```

## Results

Upon completing the steps above, you will obtain various outputs stored in the `output/` directory. These include:

### Example Outputs

- **Model Evaluation**: Key metrics such as Mean Squared Error (MSE), R-squared score (R²), and Average Absolute Relative Deviation (AARD) for each model.
- **Feature Importance**: A ranked list of features based on their importance, providing insights into which molecular descriptors drive predictions.
- **Correlation Heatmap**: A graphical representation of feature correlations, revealing potential redundancies and enabling a more refined feature selection process.

## System Requirements and Configuration

### Recommended System Specifications

The following system configuration is recommended to ensure smooth execution, especially during ChemBERTa embedding generation and model training:

- **Operating System**: Windows 10 (version 19045) or compatible Linux distribution.
- **Processor**: Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz, featuring 14 physical cores and 20 logical processors, with a maximum frequency of 3.5GHz.
- **Graphics Card**: NVIDIA GeForce RTX 4060 Ti with 16GB of dedicated VRAM, particularly beneficial for GPU-accelerated tasks.
- **Memory**: Minimum of 16GB RAM (additional memory is advantageous for handling larger datasets and intensive model training tasks).

### Additional Notes

The project leverages GPU acceleration for ChemBERTa feature extraction. Ensure that PyTorch is configured to utilize the GPU (CUDA) if an NVIDIA GPU is available, significantly improving computation speed for embedding generation and model training.

--- 

By meticulously integrating ChemBERTa embeddings and physicochemical descriptors, this project demonstrates a scalable framework for the predictive modeling of DES properties. The approach emphasizes both feature interpretability and predictive accuracy, marking a promising advancement in chemical informatics for sustainable solvent design.
```

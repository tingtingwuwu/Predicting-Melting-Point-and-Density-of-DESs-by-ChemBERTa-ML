```markdown
# Predicting Properties of Deep Eutectic Solvents (DESs) using ChemBERTa and Ensemble Models

## Abstract

Deep eutectic solvents (DESs) have emerged as promising sustainable alternatives to conventional solvents due to their low toxicity, biodegradability, and broad applicability. However, predicting crucial DES properties, such as melting point and density, is challenging due to their complex molecular structures. Traditional predictive models often struggle to capture subtle structural features from SMILES (Simplified Molecular Input Line Entry System) representations, resulting in limited accuracy.

In this study, we utilize ChemBERTa, a pre-trained Transformer model, to extract high-dimensional embeddings from SMILES strings, which effectively capture complex molecular interactions and nuanced structural characteristics. We identified certain limitations in ChemBERTa embeddings through feature importance analysis and supplemented them with additional physicochemical descriptors from RDKit. This combined feature set enhances both interpretability and predictive accuracy.

Optimized ensemble models, including ExtraTreesRegressor (ETR) and XGBRegressor (XGBR), are applied to this enriched feature set, achieving significant improvements in prediction accuracy for DES melting point and density. Rigorous grid search and ten-fold cross-validation are employed to ensure model robustness and generalizability. Our findings underscore the transformative role of pre-trained deep learning models in chemical informatics, supporting scalable, sustainable DES design.

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

The directory structure of this project is organized as follows:

```
Model architecture/
├── README.md                           # Project description and usage instructions
├── data/                               # Data folder
│   ├── __init__.py
│   ├── descriptor_extraction.py        # Extracts molecular descriptors from SMILES
│   ├── DESs_density_data.csv           # Density data for DESs
│   └── DESs_melting_point_data.csv     # Melting point data for DESs
├── output/                             # Output folder for generated files
├── src/                                # Source code
│   ├── __init__.py
│   ├── models/                         # Model scripts
│   │   ├── __init__.py
│   │   ├── chemberta_model_comparison.py
│   │   └── chemberta_model_grid_search.py
│   └── utils/                          # Utility scripts
│       ├── __init__.py
│       ├── correlation_heatmap.py
│       └── feature_importance_extraction.py
└── requirements.txt                    # Project dependencies
```

## Dependencies

To set up the environment, install the required Python packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Key Libraries

- **pandas**: For data manipulation and preprocessing.
- **torch** and **transformers**: For leveraging ChemBERTa for molecular embedding extraction.
- **scikit-learn**: For building, training, and evaluating machine learning models.
- **seaborn** and **matplotlib**: For data visualization.
- **numpy**: For efficient numerical computations.

## Data

The project uses two main datasets:

- **DESs_density_data.csv**: Contains density measurements for various DES compounds.
- **DESs_melting_point_data.csv**: Contains melting point data for the same DES compounds.

These datasets are located in the `data/` directory and are used as input for descriptor extraction and feature generation.

## Code Modules

### Data Preparation and Descriptor Extraction

- **descriptor_extraction.py**: This script generates two types of features:
  - **ChemBERTa embeddings** from SMILES strings, capturing intricate molecular patterns and relationships.
  - **RDKit descriptors** to augment ChemBERTa embeddings with additional physicochemical information for better model interpretability.
  
  The extracted features are saved for model training.

### Models

- **chemberta_model_comparison.py**: Trains multiple models (e.g., ExtraTreesRegressor, RandomForestRegressor) on the prepared dataset to predict DES properties and compares their performance.
- **chemberta_model_grid_search.py**: Conducts grid search and cross-validation to optimize hyperparameters, further improving model accuracy and robustness.

### Utilities

- **correlation_heatmap.py**: Generates a heatmap to visualize feature correlations, aiding in identifying redundant or highly correlated descriptors.
- **feature_importance_extraction.py**: Extracts and ranks features based on their importance, providing insights into which descriptors most significantly influence DES property predictions.

## Usage

### Set up the Environment

Ensure all dependencies are installed by running:

```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

#### Step 1: Data Preparation

Run `descriptor_extraction.py` to extract molecular descriptors and ChemBERTa embeddings. Make sure the input data files (`DESs_density_data.csv`, `DESs_melting_point_data.csv`) are in the `data/` directory.

```bash
python data/descriptor_extraction.py
```

#### Step 2: Model Training and Comparison

Execute `chemberta_model_comparison.py` to train and evaluate different models on the extracted features, comparing performance across models to identify the most effective predictor.

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

After following the steps above, various outputs will be stored in the `output/` directory, including:

### Example Outputs

- **Model Evaluation**: Key metrics such as Mean Squared Error (MSE), R-squared score (R²), and Average Absolute Relative Deviation (AARD) for each model.
- **Feature Importance**: A ranked list of features by importance, providing insights into which molecular descriptors most influence predictions.
- **Correlation Heatmap**: A graphical representation of feature correlations, revealing potential redundancies and aiding in refined feature selection.

## System Requirements and Configuration

### Recommended System Specifications

The following configuration is recommended for smooth execution, particularly during ChemBERTa embedding extraction and model training:

- **Operating System**: Windows 10 (version 19045) or a compatible Linux distribution.
- **Processor**: Intel(R) Core(TM) i7-11800H CPU @ 2.30GHz, with 14 physical cores and 20 logical processors, and a maximum frequency of 3.5GHz.
- **Graphics Card**: NVIDIA GeForce RTX 4060 Ti with 16GB of dedicated VRAM, especially beneficial for GPU-accelerated tasks.
- **Memory**: Minimum of 16GB RAM (additional memory is advantageous for handling larger datasets and intensive model training tasks).

### Additional Notes

The project leverages GPU acceleration for ChemBERTa feature extraction. Ensure that PyTorch is configured to utilize the GPU (CUDA) if an NVIDIA GPU is available, significantly improving computation speed for embedding generation and model training.

--- 

By combining ChemBERTa embeddings with physicochemical descriptors, this project establishes a robust framework for DES property prediction, emphasizing both feature interpretability and predictive accuracy. This approach represents a significant step forward in chemical informatics, contributing to sustainable solvent design.
```

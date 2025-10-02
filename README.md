# Drug_Gene_interaction
# Pharmacogenomics Interaction Prediction Framework

A comprehensive machine learning framework for predicting gene-drug interactions using multi-modal feature integration. This implementation combines semantic embeddings, biological sequence analysis, and literature evidence to provide research-grade predictions for pharmacogenomic relationships.

## Overview

This framework implements a complete pipeline for training and deploying machine learning models that predict interactions between genes and drugs. The system integrates multiple data modalities including:

- Semantic representations of genes and drugs using transformer-based embeddings
- DNA sequence features extracted from NCBI gene databases
- Evidence from PubMed scientific literature
- Statistical similarity metrics between molecular representations

The framework is designed for academic research and includes comprehensive evaluation metrics, cross-validation, hyperparameter optimization, and interpretability features.

## Features

### Data Integration

- Loads PharmGKB curated gene-drug interaction databases
- Fetches gene sequences from NCBI using Entrez API
- Retrieves PubMed abstracts for evidence-based predictions
- Implements caching to minimize redundant API calls

### Feature Engineering

- Multi-modal feature extraction combining semantic, sequence, and statistical information
- Sentence transformer embeddings for genes, drugs, and evidence text
- Biological sequence features: GC content, nucleotide composition, molecular weight
- Interaction features computed from element-wise products
- Cosine similarity and Euclidean distance metrics

### Model Training

- Six baseline models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- Bayesian hyperparameter optimization using Optuna
- Stratified k-fold cross-validation
- SMOTE-Tomek resampling for class imbalance
- Comprehensive evaluation metrics: AUC-ROC, F1-score, MCC, Cohen's Kappa

### Deployment Tools

- Interactive web interface for single and batch predictions
- Training pipeline with real-time progress monitoring
- Model artifacts export for production deployment
- Clinical interpretation guidance for predictions

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum, 16GB recommended
- Internet connection for API access to NCBI and PubMed

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

### NCBI Configuration

Before running the framework, you must configure your NCBI credentials. NCBI requires a valid email address for API access.

Edit `gene_drug_framework.py` line 145:

```python
Entrez.email = "your.email@institution.edu"  # Required by NCBI
```

Optionally, obtain an API key from https://www.ncbi.nlm.nih.gov/account/ for faster access:

```python
Entrez.api_key = "your_api_key_here"  # Optional but recommended
```

## Data Preparation

### Required Data Files

Place your PharmGKB data files in the `data/` directory:

```
data/
├── genes.tsv
├── drugs.tsv
└── relationships.tsv
```

These files should be downloaded from PharmGKB (https://www.pharmgkb.org/downloads). The framework expects tab-separated values with standard PharmGKB column headers.

### Data Format

The framework automatically detects column names, but expects:

**genes.tsv**: Gene identifiers and names
**drugs.tsv**: Drug identifiers and names
**relationships.tsv**: Gene-drug associations with evidence

## Usage

### Command Line Training

Run the complete training pipeline:

```bash
python gene_drug_framework.py \
    --data-dir data \
    --output-dir outputs \
    --seed 42 \
    --cv-folds 10 \
    --n-trials 100
```

Options:
- `--data-dir`: Directory containing PharmGKB TSV files
- `--output-dir`: Output directory for models and results
- `--seed`: Random seed for reproducibility
- `--cv-folds`: Number of cross-validation folds
- `--n-trials`: Optuna optimization trials
- `--no-sequences`: Disable gene sequence fetching
- `--no-pubmed`: Disable PubMed abstract fetching

### Interactive Training Pipeline

Launch the Streamlit training interface:

```bash
streamlit run streamlit.py
```

This provides a web interface for:
- Configuring experiment parameters
- Monitoring training progress
- Visualizing results
- Exporting trained models

### Prediction Interface

Launch the prediction web application:

```bash
streamlit run app.py
```

Features:
- Single gene-drug pair prediction
- Batch analysis from CSV files
- Feature importance visualization
- Clinical interpretation guidance

### Python API

Use the framework programmatically:

```python
from gene_drug_framework import ExperimentConfig, run_complete_pipeline
from Bio import Entrez

# Configure
config = ExperimentConfig()
config.SEED = 42
config.FETCH_GENE_SEQUENCES = True
config.FETCH_PUBMED_ABSTRACTS = True

# Set email (required)
Entrez.email = "your.email@institution.edu"

# Run pipeline
results = run_complete_pipeline(config)

# Access results
print(f"Test AUC: {results['test_metrics']['auc_macro']:.4f}")
print(f"Model saved to: {results['model_path']}")
```

## Configuration

### Key Parameters

Edit `gene_drug_framework.py` or use command-line arguments:

```python
# Reproducibility
SEED = 42

# Embedding model
EMBEDDING_MODEL = 'all-mpnet-base-v2'

# Data splitting
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Cross-validation
CV_FOLDS = 10

# Optimization
N_OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # seconds

# Resampling strategy
RESAMPLING_STRATEGY = 'SMOTE-Tomek'

# Feature engineering
USE_SEQUENCE_FEATURES = True
USE_STATISTICAL_FEATURES = True
FETCH_GENE_SEQUENCES = True
FETCH_PUBMED_ABSTRACTS = True
```

## Output Structure

The framework generates organized output:

```
outputs/
├── pharmacogenomics_experiment_TIMESTAMP/
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   ├── feature_engineer.pkl
│   │   └── metadata.json
│   ├── results/
│   │   ├── complete_results.json
│   │   ├── cross_validation_results.json
│   │   ├── test_metrics.json
│   │   └── CatBoost_optimization.json
│   ├── figures/
│   └── experiment_config.json
```

## Model Performance

Typical performance on PharmGKB data:

- AUC-ROC: 0.85-0.90
- F1-Score: 0.80-0.85
- Accuracy: 0.78-0.83
- Matthews Correlation Coefficient: 0.65-0.75

Performance varies based on dataset size, class balance, and feature configuration.

## Prediction Classes

The model predicts three classes:

1. **Associated (1)**: Strong evidence of pharmacogenomic interaction
2. **Not Associated (0)**: No significant interaction detected
3. **Ambiguous (2)**: Conflicting or insufficient evidence

Each prediction includes probability scores and confidence metrics.

## Clinical Interpretation

Predictions should be interpreted with caution:

- High confidence (greater than 85 percent): Strong model confidence
- Moderate confidence (70-85 percent): Review with additional evidence
- Low confidence (less than 70 percent): Requires validation

Always consult clinical pharmacogenomics guidelines (CPIC, FDA) and consider patient-specific factors.

## API Rate Limits

### NCBI/PubMed

Without API key: 3 requests per second
With API key: 10 requests per second

The framework implements automatic rate limiting. For large datasets, consider obtaining an API key to reduce processing time.

## Troubleshooting

### Common Issues

**ImportError for Bio or other packages**
```bash
pip install biopython sentence-transformers xgboost catboost lightgbm
```

**NCBI API errors**
- Verify email is set in Entrez.email
- Check internet connection
- Consider adding API key for higher limits

**Memory errors during training**
- Reduce batch size in configuration
- Disable sequence or PubMed fetching if not needed
- Use a machine with more RAM

**Poor model performance**
- Check class balance in dataset
- Verify TSV files are correctly formatted
- Try different resampling strategies
- Increase number of Optuna trials

## Limitations

This framework has several important limitations:

1. **Training data dependency**: Limited to interactions present in PharmGKB
2. **Population bias**: May not generalize across diverse populations
3. **Temporal limitations**: Requires periodic retraining as evidence evolves
4. **Computational requirements**: Full pipeline requires significant compute time
5. **API dependencies**: Requires internet access for sequence and literature data

## Research Use Only

This tool is intended for research and educational purposes only. It is not:

- FDA approved for clinical use
- A substitute for genetic testing
- A replacement for professional medical judgment
- Validated for all gene-drug combinations

Clinical decisions should integrate multiple sources of evidence and follow established pharmacogenomics guidelines.

## Citation

If you use this framework in your research, please cite:

```
[Pending publication]
Title: Deep Learning-Enhanced Pharmacogenomic Interaction Prediction
Authors: [Authors]
Journal: [Journal], Year
DOI: [DOI]
```

## License

This project is released under the MIT License for academic and research use.

## Contributing

Contributions are welcome. Please submit issues or pull requests on the project repository.

## Contact

For questions, bug reports, or collaboration:

- Email: [faaiz.ds24@duk.ac.in]
- GitHub: [repository URL]
- Issues: [GitHub issues URL]

## Acknowledgments

This framework uses data from:
- PharmGKB (www.pharmgkb.org)
- NCBI Gene (www.ncbi.nlm.nih.gov/gene)
- PubMed (pubmed.ncbi.nlm.nih.gov)

Built with:
- Sentence Transformers for semantic embeddings
- BioPython for sequence analysis
- Scikit-learn for machine learning
- XGBoost, CatBoost, LightGBM for gradient boosting
- Optuna for hyperparameter optimization
- Streamlit for web interfaces

## Version History

### Version 1.0.0
- Initial release
- Multi-modal feature engineering
- Six baseline models with optimization
- Web interfaces for training and prediction
- Comprehensive evaluation metrics

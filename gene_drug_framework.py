"""
==============================================================================
PHARMACOGENOMIC INTERACTION PREDICTION FRAMEWORK
Research-Grade Implementation for Academic Publication
==============================================================================

Title: Deep Learning-Enhanced Pharmacogenomic Interaction Prediction Using 
       Multi-Modal Feature Integration and Ensemble Learning

Authors: [Your Name et al.]
Institution: [Your Institution]
Contact: [email]

This implementation provides a complete, reproducible framework for predicting
gene-drug interactions using advanced machine learning and natural language
processing techniques.

Citation: [To be added upon publication]
License: MIT (Academic Use)
==============================================================================
"""

import os
import sys
import json
import time
import random
import warnings
import shelve
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import joblib

# Scientific computing
from scipy import stats
from scipy.spatial.distance import pdist, squareform

# Machine learning
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_score
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    precision_recall_curve, f1_score, accuracy_score,
    roc_curve, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Advanced ML models
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek

# NLP and biological data
from sentence_transformers import SentenceTransformer
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, gc_fraction

# Interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pharmacogenomics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# ==============================================================================

class ExperimentConfig:
    """
    Central configuration for reproducible experiments.
    All hyperparameters and settings for publication-quality results.
    """
    
    def __init__(self):
        # Reproducibility
        self.SEED = 42
        self.EXPERIMENT_NAME = f"pharmacogenomics_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.set_seeds()
        
        # Data paths
        self.DATA_DIR = Path("data")
        self.GENES_TSV = self.DATA_DIR / "genes.tsv"
        self.DRUGS_TSV = self.DATA_DIR / "drugs.tsv"
        self.RELS_TSV = self.DATA_DIR / "relationships.tsv"
        
        # Output directories
        self.OUTPUT_DIR = Path("outputs") / self.EXPERIMENT_NAME
        self.MODEL_DIR = self.OUTPUT_DIR / "models"
        self.RESULTS_DIR = self.OUTPUT_DIR / "results"
        self.FIGURES_DIR = self.OUTPUT_DIR / "figures"
        self.CACHE_DIR = Path("cache")
        
        # Create directories
        for directory in [self.OUTPUT_DIR, self.MODEL_DIR, self.RESULTS_DIR, 
                         self.FIGURES_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.PMID_CACHE = self.CACHE_DIR / "pmid_cache.db"
        self.GENE_SEQ_CACHE = self.CACHE_DIR / "gene_seq_cache.json"
        
        # Computational resources
        self.DEVICE = self._detect_device()
        self.N_JOBS = -1  # Use all available cores
        
        # Embedding configuration
        self.EMBEDDING_MODEL = 'all-mpnet-base-v2'
        self.EMBEDDING_DIM = 768  # For all-mpnet-base-v2
        self.BATCH_SIZE = 32
        
        # Biological data fetching
        self.FETCH_GENE_SEQUENCES = True
        self.FETCH_PUBMED_ABSTRACTS = True
        self.MAX_ABSTRACT_LENGTH = 5000
        self.PUBMED_BATCH_SIZE = 50
        self.PUBMED_RATE_LIMIT = 0.34  # Seconds between requests (3 req/sec)
        
        # Data splitting strategy
        self.TEST_SIZE = 0.15
        self.VAL_SIZE = 0.15
        self.STRATIFY = True
        
        # Resampling strategy
        self.RESAMPLING_STRATEGY = 'SMOTE-Tomek'  # Options: 'SMOTE', 'ADASYN', 'SMOTE-Tomek', 'BorderlineSMOTE'
        self.SMOTE_K_NEIGHBORS = 5
        
        # Model training
        self.N_OPTUNA_TRIALS = 100
        self.OPTUNA_TIMEOUT = 3600  # 1 hour
        self.CV_FOLDS = 10
        self.EARLY_STOPPING_ROUNDS = 50
        
        # Feature engineering
        self.USE_SEQUENCE_FEATURES = True
        self.USE_STATISTICAL_FEATURES = True
        self.NORMALIZE_FEATURES = True
        self.SCALER_TYPE = 'robust'  # Options: 'standard', 'robust'
        
        # Interpretability
        self.SHAP_SAMPLES = 500
        self.PERMUTATION_REPEATS = 10
        self.PERMUTATION_N_SAMPLES = 1000
        
        # Model ensemble
        self.USE_ENSEMBLE = True
        self.ENSEMBLE_VOTING = 'soft'  # Options: 'hard', 'soft'
        
        # Entrez configuration - CHANGE THESE
        Entrez.email = "faaiz,ds242duk.ac.in"  # REQUIRED: Change to your email
        Entrez.api_key = "1ed2a4f8626f38b2bccb05b499bc8ad54009"  # Add NCBI API key for faster access (optional)
        
        # Save configuration
        self.save_config()
        
        logger.info(f"Initialized experiment: {self.EXPERIMENT_NAME}")
        logger.info(f"Using device: {self.DEVICE}")
    
    def _detect_device(self) -> str:
        """Detect available computational device"""
        try:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
        except ImportError:
            device = 'cpu'
        return device
    
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        os.environ['PYTHONHASHSEED'] = str(self.SEED)
        
        try:
            import torch
            torch.manual_seed(self.SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def save_config(self):
        """Save configuration to JSON"""
        config_dict = {
            'experiment_name': self.EXPERIMENT_NAME,
            'seed': self.SEED,
            'device': self.DEVICE,
            'embedding_model': self.EMBEDDING_MODEL,
            'test_size': self.TEST_SIZE,
            'val_size': self.VAL_SIZE,
            'cv_folds': self.CV_FOLDS,
            'n_optuna_trials': self.N_OPTUNA_TRIALS,
            'resampling_strategy': self.RESAMPLING_STRATEGY,
            'fetch_sequences': self.FETCH_GENE_SEQUENCES,
            'fetch_abstracts': self.FETCH_PUBMED_ABSTRACTS,
        }
        
        config_path = self.OUTPUT_DIR / 'experiment_config.json'
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Configuration saved to {config_path}")

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class PerformanceMetrics:
    """Comprehensive model performance metrics for reporting"""
    model_name: str
    auc_macro: float
    auc_weighted: float
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_micro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    mcc: float
    cohen_kappa: float
    confusion_matrix: np.ndarray
    classification_report: Dict
    per_class_metrics: Dict
    
    def to_dict(self):
        d = asdict(self)
        d['confusion_matrix'] = self.confusion_matrix.tolist()
        return d
    
    def save(self, path: Path):
        """Save metrics to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

@dataclass
class CrossValidationResults:
    """Cross-validation results for statistical reporting"""
    model_name: str
    cv_scores: Dict[str, np.ndarray]
    mean_scores: Dict[str, float]
    std_scores: Dict[str, float]
    fold_predictions: List[np.ndarray]
    
    def get_confidence_interval(self, metric: str, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for a metric"""
        scores = self.cv_scores[metric]
        mean = np.mean(scores)
        sem = stats.sem(scores)
        ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=sem)
        return ci

# ==============================================================================
# DATA LOADING AND PREPROCESSING
# ==============================================================================

class PharmGKBDataLoader:
    """
    Load and preprocess PharmGKB data with comprehensive validation.
    Implements quality control and data integrity checks.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.genes_df = None
        self.drugs_df = None
        self.relationships_df = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all PharmGKB TSV files with validation"""
        logger.info("Loading PharmGKB data files...")
        
        def safe_read_tsv(filepath: Path) -> pd.DataFrame:
            """Read TSV with multiple encoding attempts"""
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(
                        filepath, 
                        sep='\t', 
                        dtype=str, 
                        on_bad_lines='skip',
                        engine='python',
                        encoding=encoding
                    )
                    logger.info(f"Successfully loaded {filepath.name} with {encoding} encoding")
                    return df
                except (UnicodeDecodeError, Exception) as e:
                    continue
            raise IOError(f"Could not read {filepath} with any encoding")
        
        # Load files
        self.genes_df = safe_read_tsv(self.config.GENES_TSV)
        self.drugs_df = safe_read_tsv(self.config.DRUGS_TSV)
        self.relationships_df = safe_read_tsv(self.config.RELS_TSV)
        
        # Clean string columns
        for df in [self.genes_df, self.drugs_df, self.relationships_df]:
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].str.strip()
        
        # Log data statistics
        logger.info(f"Loaded {len(self.genes_df)} genes")
        logger.info(f"Loaded {len(self.drugs_df)} drugs")
        logger.info(f"Loaded {len(self.relationships_df)} relationships")
        
        # Data quality report
        self._log_data_quality()
        
        return self.genes_df, self.drugs_df, self.relationships_df
    
    def _log_data_quality(self):
        """Log data quality metrics"""
        logger.info("\nData Quality Report:")
        logger.info(f"Genes - Missing values: {self.genes_df.isnull().sum().sum()}")
        logger.info(f"Drugs - Missing values: {self.drugs_df.isnull().sum().sum()}")
        logger.info(f"Relationships - Missing values: {self.relationships_df.isnull().sum().sum()}")
        
        # Check for duplicates
        logger.info(f"Duplicate genes: {self.genes_df.duplicated().sum()}")
        logger.info(f"Duplicate drugs: {self.drugs_df.duplicated().sum()}")
    
    def create_interaction_dataset(self) -> pd.DataFrame:
        """
        Create comprehensive gene-drug interaction dataset.
        Implements label mapping and data filtering.
        """
        logger.info("\nCreating gene-drug interaction dataset...")
        
        # Find columns (flexible column naming)
        def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                if candidate in df.columns:
                    return candidate
            return None
        
        # Identify key columns
        gene_id_col = find_column(self.genes_df, ['PharmGKB Accession Id', 'Accession Id', 'Gene ID'])
        gene_name_col = find_column(self.genes_df, ['Name', 'Gene Name', 'Symbol'])
        gene_symbol_col = find_column(self.genes_df, ['Symbol', 'Gene Symbol'])
        
        drug_id_col = find_column(self.drugs_df, ['PharmGKB Accession Id', 'Accession Id', 'Drug ID'])
        drug_name_col = find_column(self.drugs_df, ['Name', 'Drug Name'])
        
        e1_id = find_column(self.relationships_df, ['Entity1_id', 'Gene ID'])
        e1_type = find_column(self.relationships_df, ['Entity1_type', 'Entity Type'])
        e2_id = find_column(self.relationships_df, ['Entity2_id', 'Chemical ID'])
        e2_type = find_column(self.relationships_df, ['Entity2_type', 'Entity Type'])
        assoc_col = find_column(self.relationships_df, ['Association', 'Association Type'])
        pmids_col = find_column(self.relationships_df, ['PMIDs', 'PubMed IDs'])
        evidence_col = find_column(self.relationships_df, ['Evidence', 'Evidence Level'])
        
        # Validate required columns
        required_cols = [gene_id_col, gene_name_col, drug_id_col, drug_name_col,
                        e1_id, e1_type, e2_id, e2_type, assoc_col]
        if not all(required_cols):
            missing = [name for name, col in zip(
                ['gene_id', 'gene_name', 'drug_id', 'drug_name', 'e1_id', 'e1_type', 'e2_id', 'e2_type', 'association'],
                required_cols
            ) if col is None]
            raise ValueError(f"Missing required columns: {missing}")
        
        # Filter for gene-drug relationships
        mask = (
            (self.relationships_df[e1_type].str.lower() == 'gene') &
            (self.relationships_df[e2_type].str.lower().isin(['chemical', 'drug']))
        )
        gd_rels = self.relationships_df[mask].copy()
        gd_rels = gd_rels.rename(columns={e1_id: 'gene_id', e2_id: 'drug_id'})
        
        # Label mapping with validation
        def map_association_label(text: str) -> int:
            """
            Map association text to numerical label.
            0 = Not Associated
            1 = Associated  
            2 = Ambiguous
            -1 = Unknown/Invalid
            """
            text = str(text).lower()
            if 'not associated' in text or 'no association' in text:
                return 0
            elif 'associated' in text and 'not associated' not in text:
                return 1
            elif 'ambiguous' in text or 'unclear' in text:
                return 2
            else:
                return -1
        
        gd_rels['label'] = gd_rels[assoc_col].apply(map_association_label)
        
        # Filter valid labels
        valid_mask = gd_rels['label'].isin([0, 1, 2])
        pairs_df = gd_rels[valid_mask].copy()
        
        logger.info(f"Filtered to {len(pairs_df)} valid gene-drug pairs")
        logger.info(f"Excluded {(~valid_mask).sum()} invalid relationships")
        
        # Map gene and drug names
        gene_map = self.genes_df.set_index(gene_id_col)[gene_name_col].to_dict()
        drug_map = self.drugs_df.set_index(drug_id_col)[drug_name_col].to_dict()
        
        pairs_df['gene_name'] = pairs_df['gene_id'].map(gene_map)
        pairs_df['drug_name'] = pairs_df['drug_id'].map(drug_map)
        
        # Add gene symbols if available
        if gene_symbol_col:
            gene_symbol_map = self.genes_df.set_index(gene_id_col)[gene_symbol_col].to_dict()
            pairs_df['gene_symbol'] = pairs_df['gene_id'].map(gene_symbol_map)
        
        # Add PMIDs
        if pmids_col:
            pmid_aggregation = gd_rels.groupby(['gene_id', 'drug_id'])[pmids_col].apply(
                lambda x: ';'.join(x.dropna().astype(str))
            ).to_dict()
            pairs_df['PMIDs'] = pairs_df.apply(
                lambda row: pmid_aggregation.get((row['gene_id'], row['drug_id']), ''),
                axis=1
            )
        else:
            pairs_df['PMIDs'] = ''
        
        # Add evidence if available
        if evidence_col:
            evidence_aggregation = gd_rels.groupby(['gene_id', 'drug_id'])[evidence_col].apply(
                lambda x: ';'.join(x.dropna().astype(str))
            ).to_dict()
            pairs_df['Evidence'] = pairs_df.apply(
                lambda row: evidence_aggregation.get((row['gene_id'], row['drug_id']), ''),
                axis=1
            )
        
        pairs_df['Association'] = pairs_df[assoc_col]
        
        # Remove entries with missing critical information
        pairs_df.dropna(subset=['gene_name', 'drug_name'], inplace=True)
        
        # Shuffle for randomness
        pairs_df = pairs_df.sample(frac=1, random_state=self.config.SEED).reset_index(drop=True)
        
        # Log label distribution
        logger.info("\nLabel Distribution:")
        label_names = {0: 'Not Associated', 1: 'Associated', 2: 'Ambiguous'}
        for label, count in pairs_df['label'].value_counts().sort_index().items():
            percentage = (count / len(pairs_df)) * 100
            logger.info(f"  {label_names[label]} ({label}): {count} ({percentage:.2f}%)")
        
        # Save dataset
        output_path = self.config.OUTPUT_DIR / 'interaction_dataset.csv'
        pairs_df.to_csv(output_path, index=False)
        logger.info(f"\nDataset saved to {output_path}")
        
        return pairs_df

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================

class MultiModalFeatureEngineer:
    """
    Advanced feature engineering combining multiple modalities:
    - Semantic embeddings (gene, drug, text)
    - Sequence features (DNA properties)
    - Statistical features (similarity metrics)
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.embedder = None
        self.gene_features_cache = {}
        
    def initialize_embedder(self):
        """Initialize sentence transformer model"""
        if self.embedder is None:
            logger.info(f"Loading embedding model: {self.config.EMBEDDING_MODEL}")
            self.embedder = SentenceTransformer(
                self.config.EMBEDDING_MODEL,
                device=self.config.DEVICE
            )
            embedding_dim = self.embedder.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {embedding_dim}")
            
            if embedding_dim != self.config.EMBEDDING_DIM:
                logger.warning(f"Expected dim {self.config.EMBEDDING_DIM}, got {embedding_dim}")
    
    def parse_pmids(self, pmid_string: str) -> List[str]:
        """Extract valid PMIDs from string"""
        if not isinstance(pmid_string, str) or pmid_string == 'nan' or not pmid_string:
            return []
        
        pmids = []
        for p in pmid_string.replace(';', ',').split(','):
            p = p.strip()
            # Validate PMID format (typically 7-8 digits)
            if p.isdigit() and 7 <= len(p) <= 8:
                pmids.append(p)
        
        return list(set(pmids))  # Remove duplicates
    
    def fetch_pubmed_abstracts(self, pmids: List[str]):
        """
        Fetch PubMed abstracts with caching and rate limiting.
        Implements robust error handling for API failures.
        """
        if not self.config.FETCH_PUBMED_ABSTRACTS or not pmids:
            return
        
        with shelve.open(str(self.config.PMID_CACHE)) as cache:
            to_fetch = [p for p in pmids if p not in cache]
            
            if not to_fetch:
                return
            
            logger.info(f"Fetching {len(to_fetch)} PubMed abstracts...")
            batch_size = self.config.PUBMED_BATCH_SIZE
            
            for i in tqdm(range(0, len(to_fetch), batch_size), desc="Fetching PubMed"):
                batch = to_fetch[i:i+batch_size]
                
                try:
                    # Fetch batch from PubMed
                    handle = Entrez.efetch(
                        db="pubmed",
                        id=",".join(batch),
                        rettype="xml",
                        retmode="xml"
                    )
                    records = Entrez.read(handle)
                    handle.close()
                    
                    # Process records
                    for rec in records.get('PubmedArticle', []):
                        pmid = str(rec.get('MedlineCitation', {}).get('PMID', ''))
                        if not pmid:
                            continue
                        
                        article = rec.get('MedlineCitation', {}).get('Article', {})
                        title = str(article.get('ArticleTitle', ''))
                        
                        abstract_node = article.get('Abstract', {})
                        abstract_texts = abstract_node.get('AbstractText', [])
                        
                        if isinstance(abstract_texts, list):
                            abstract = " ".join([str(t) for t in abstract_texts])
                        else:
                            abstract = str(abstract_texts)
                        
                        full_text = f"{title} {abstract}".strip()
                        
                        # Truncate if too long
                        if len(full_text) > self.config.MAX_ABSTRACT_LENGTH:
                            full_text = full_text[:self.config.MAX_ABSTRACT_LENGTH]
                        
                        cache[pmid] = full_text
                    
                    # Rate limiting
                    time.sleep(self.config.PUBMED_RATE_LIMIT if not Entrez.api_key else 0.1)
                    
                except Exception as e:
                    logger.warning(f"Error fetching PubMed batch: {e}")
                    time.sleep(2)  # Back off on error
    
    def fetch_gene_sequence(self, gene_name: str, gene_symbol: str = None) -> Optional[str]:
        """
        Fetch gene sequence from NCBI with caching.
        Returns DNA sequence for feature extraction.
        """
        if not self.config.FETCH_GENE_SEQUENCES:
            return None
        
        # Check cache
        cache_path = self.config.GENE_SEQ_CACHE
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                cache = json.load(f)
                if gene_name in cache:
                    return cache[gene_name]
        else:
            cache = {}
        
        # Search for gene
        search_term = gene_symbol if gene_symbol else gene_name
        search_query = f"{search_term}[Gene Name] AND Homo sapiens[Organism]"
        
        try:
            # Search gene database
            handle = Entrez.esearch(db="gene", term=search_query, retmax=1)
            record = Entrez.read(handle)
            handle.close()
            
            if record['IdList']:
                gene_id = record['IdList'][0]
                
                # Fetch sequence
                handle = Entrez.efetch(
                    db="gene",
                    id=gene_id,
                    rettype="fasta_cds_na",
                    retmode="text"
                )
                sequence = handle.read()
                handle.close()
                
                if sequence:
                    cache[gene_name] = sequence
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f, indent=2)
                    return sequence
        
        except Exception as e:
            logger.debug(f"Could not fetch sequence for {gene_name}: {e}")
        
        return None
    
    def compute_sequence_features(self, sequence: str) -> Dict[str, float]:
        """
        Extract DNA sequence features for modeling.
        
        Features:
        - Length (normalized)
        - GC content
        - Molecular weight (normalized)
        - Nucleotide composition (A, T, G, C)
        """
        try:
            # Remove FASTA header
            if sequence.startswith('>'):
                sequence = ''.join(sequence.split('\n')[1:])
            
            # Create Seq object
            seq = Seq(sequence.upper().replace('U', 'T'))
            seq_len = len(seq)
            
            if seq_len == 0:
                raise ValueError("Empty sequence")
            
            features = {
                'seq_length': seq_len / 10000.0,  # Normalize to typical gene length
                'gc_content': gc_fraction(seq),
                'molecular_weight': molecular_weight(seq, seq_type='DNA') / 100000.0,
                'a_content': seq.count('A') / seq_len,
                't_content': seq.count('T') / seq_len,
                'g_content': seq.count('G') / seq_len,
                'c_content': seq.count('C') / seq_len,
            }
            
            return features
        
        except Exception as e:
            logger.debug(f"Error computing sequence features: {e}")
            # Return zero features on error
            return {
                'seq_length': 0.0,
                'gc_content': 0.0,
                'molecular_weight': 0.0,
                'a_content': 0.0,
                't_content': 0.0,
                'g_content': 0.0,
                'c_content': 0.0
            }
    
    def create_evidence_text(self, row: pd.Series) -> str:
        """
        Create evidence text from multiple sources:
        - Gene information
        - Drug information
        - Association type
        - PubMed abstracts (if available)
        """
        pmids = self.parse_pmids(row['PMIDs'])
        
        # If no PMIDs or not fetching, create structured text
        if not pmids or not self.config.FETCH_PUBMED_ABSTRACTS:
            parts = [
                f"Gene: {row['gene_name']}",
                f"Drug: {row['drug_name']}",
                f"Association: {row['Association']}"
            ]
            
            if 'gene_symbol' in row and pd.notna(row['gene_symbol']):
                parts.append(f"Gene Symbol: {row['gene_symbol']}")
            
            if 'Evidence' in row and pd.notna(row['Evidence']):
                parts.append(f"Evidence: {row['Evidence']}")
            
            return ". ".join(parts)
        
        # Fetch and combine abstracts
        self.fetch_pubmed_abstracts(pmids)
        
        with shelve.open(str(self.config.PMID_CACHE), 'r') as cache:
            texts = [cache.get(p, "") for p in pmids if p in cache]
        
        if texts:
            return " ".join(texts).strip()
        else:
            return f"Pharmacogenomic relationship between gene {row['gene_name']} and drug {row['drug_name']}."
    
    def create_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create comprehensive multi-modal feature matrix.
        
        Feature components:
        1. Gene embeddings (semantic)
        2. Drug embeddings (semantic)
        3. Evidence text embeddings (contextual)
        4. Interaction features (element-wise products)
        5. Sequence features (biological properties)
        6. Statistical features (similarity metrics)
        
        Returns:
            Feature matrix (n_samples, n_features) and metadata dictionary
        """
        logger.info("\nCreating multi-modal features...")
        self.initialize_embedder()
        
        embedding_dim = self.embedder.get_sentence_embedding_dimension()
        feature_components = []
        
        # 1. Gene embeddings
        logger.info("Step 1/6: Creating gene embeddings...")
        gene_embeddings = {}
        unique_genes = data[['gene_id', 'gene_name']].drop_duplicates()
        
        for _, row in tqdm(unique_genes.iterrows(), total=len(unique_genes), desc="Gene embeddings"):
            text = row['gene_name']
            
            # Add gene symbol if available
            if 'gene_symbol' in data.columns:
                gene_rows = data[data['gene_id'] == row['gene_id']]
                if len(gene_rows) > 0:
                    symbol = gene_rows['gene_symbol'].iloc[0]
                    if pd.notna(symbol):
                        text = f"{text} {symbol}"
            
            embedding = self.embedder.encode(text, show_progress_bar=False)
            gene_embeddings[row['gene_id']] = embedding
        
        # 2. Drug embeddings
        logger.info("Step 2/6: Creating drug embeddings...")
        drug_embeddings = {}
        unique_drugs = data[['drug_id', 'drug_name']].drop_duplicates()
        
        for _, row in tqdm(unique_drugs.iterrows(), total=len(unique_drugs), desc="Drug embeddings"):
            embedding = self.embedder.encode(row['drug_name'], show_progress_bar=False)
            drug_embeddings[row['drug_id']] = embedding
        
        # 3. Gene sequence features
        logger.info("Step 3/6: Processing gene sequences...")
        if self.config.USE_SEQUENCE_FEATURES:
            for _, row in tqdm(unique_genes.iterrows(), total=len(unique_genes), desc="Gene sequences"):
                gene_symbol = None
                if 'gene_symbol' in data.columns:
                    gene_rows = data[data['gene_id'] == row['gene_id']]
                    if len(gene_rows) > 0:
                        gene_symbol = gene_rows['gene_symbol'].iloc[0]
                
                sequence = self.fetch_gene_sequence(row['gene_name'], gene_symbol)
                self.gene_features_cache[row['gene_id']] = self.compute_sequence_features(sequence) if sequence else {
                    'seq_length': 0.0, 'gc_content': 0.0, 'molecular_weight': 0.0,
                    'a_content': 0.0, 't_content': 0.0, 'g_content': 0.0, 'c_content': 0.0
                }
        
        # 4. Evidence text embeddings
        logger.info("Step 4/6: Creating evidence text embeddings...")
        data['evidence_text'] = data.apply(self.create_evidence_text, axis=1)
        
        text_embeddings = self.embedder.encode(
            data['evidence_text'].tolist(),
            show_progress_bar=True,
            batch_size=self.config.BATCH_SIZE
        )
        
        # 5. Combine base embeddings
        logger.info("Step 5/6: Combining embeddings...")
        X_gene = np.array([gene_embeddings.get(gid, np.zeros(embedding_dim)) for gid in data['gene_id']])
        X_drug = np.array([drug_embeddings.get(did, np.zeros(embedding_dim)) for did in data['drug_id']])
        X_text = text_embeddings
        
        # Interaction features (element-wise products capture feature interactions)
        X_gene_text = X_gene * X_text
        X_drug_text = X_drug * X_text
        X_gene_drug = X_gene * X_drug
        
        feature_components.extend([X_gene, X_drug, X_text, X_gene_text, X_drug_text, X_gene_drug])
        
        # 6. Additional features
        logger.info("Step 6/6: Computing additional features...")
        
        # Sequence features
        if self.config.USE_SEQUENCE_FEATURES:
            X_seq = np.array([
                list(self.gene_features_cache.get(gid, {
                    'seq_length': 0.0, 'gc_content': 0.0, 'molecular_weight': 0.0,
                    'a_content': 0.0, 't_content': 0.0, 'g_content': 0.0, 'c_content': 0.0
                }).values())
                for gid in data['gene_id']
            ])
            feature_components.append(X_seq)
        
        # Statistical features
        if self.config.USE_STATISTICAL_FEATURES:
            X_stats = []
            for i in range(len(data)):
                gene_vec = X_gene[i]
                drug_vec = X_drug[i]
                
                # Compute various similarity metrics
                cosine_sim = np.dot(gene_vec, drug_vec) / (
                    np.linalg.norm(gene_vec) * np.linalg.norm(drug_vec) + 1e-10
                )
                euclidean_dist = np.linalg.norm(gene_vec - drug_vec)
                
                stats = [
                    np.mean(gene_vec), np.std(gene_vec), np.max(gene_vec), np.min(gene_vec),
                    np.mean(drug_vec), np.std(drug_vec), np.max(drug_vec), np.min(drug_vec),
                    cosine_sim, euclidean_dist
                ]
                X_stats.append(stats)
            
            X_stats = np.array(X_stats)
            feature_components.append(X_stats)
        
        # Combine all features
        X_combined = np.concatenate(feature_components, axis=1)
        
        # Create metadata
        metadata = {
            'feature_info': {
                'total_features': X_combined.shape[1],
                'gene_embedding_dim': embedding_dim,
                'drug_embedding_dim': embedding_dim,
                'text_embedding_dim': embedding_dim,
                'interaction_dims': embedding_dim * 3,
                'sequence_features': 7 if self.config.USE_SEQUENCE_FEATURES else 0,
                'statistical_features': 10 if self.config.USE_STATISTICAL_FEATURES else 0,
                'embedding_model': self.config.EMBEDDING_MODEL
            },
            'gene_embeddings': gene_embeddings,
            'drug_embeddings': drug_embeddings,
            'data': data
        }
        
        logger.info(f"\nFeature matrix created: {X_combined.shape}")
        logger.info(f"  Gene embeddings: {embedding_dim}")
        logger.info(f"  Drug embeddings: {embedding_dim}")
        logger.info(f"  Text embeddings: {embedding_dim}")
        logger.info(f"  Interaction features: {embedding_dim * 3}")
        if self.config.USE_SEQUENCE_FEATURES:
            logger.info(f"  Sequence features: 7")
        if self.config.USE_STATISTICAL_FEATURES:
            logger.info(f"  Statistical features: 10")
        
        # Save features for reproducibility
        feature_path = self.config.CACHE_DIR / 'features.npz'
        np.savez_compressed(feature_path, features=X_combined)
        logger.info(f"Features cached to {feature_path}")
        
        return X_combined, metadata

# ==============================================================================
# MODEL TRAINING AND EVALUATION
# ==============================================================================

class ModelTrainer:
    """
    Train and evaluate machine learning models with comprehensive metrics.
    Implements hyperparameter optimization and cross-validation.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.scaler = None
        self.best_model = None
        self.models = {}
        self.results = {}
    
    def prepare_data_splits(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Create stratified train/validation/test splits with resampling.
        """
        logger.info("\nPreparing data splits...")
        
        # Initial train/temp split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(self.config.TEST_SIZE + self.config.VAL_SIZE),
            random_state=self.config.SEED,
            stratify=y if self.config.STRATIFY else None
        )
        
        # Val/test split
        val_ratio = self.config.VAL_SIZE / (self.config.TEST_SIZE + self.config.VAL_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            random_state=self.config.SEED,
            stratify=y_temp if self.config.STRATIFY else None
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        # Log class distribution
        logger.info("\nTrain set distribution:")
        for label, count in Counter(y_train).items():
            logger.info(f"  Class {label}: {count} ({count/len(y_train)*100:.1f}%)")
        
        # Apply resampling to training data
        logger.info(f"\nApplying {self.config.RESAMPLING_STRATEGY}...")
        
        k_neighbors = min(self.config.SMOTE_K_NEIGHBORS, min(np.bincount(y_train)) - 1)
        
        if self.config.RESAMPLING_STRATEGY == 'SMOTE':
            resampler = SMOTE(random_state=self.config.SEED, k_neighbors=k_neighbors)
        elif self.config.RESAMPLING_STRATEGY == 'ADASYN':
            resampler = ADASYN(random_state=self.config.SEED, n_neighbors=k_neighbors)
        elif self.config.RESAMPLING_STRATEGY == 'SMOTE-Tomek':
            resampler = SMOTETomek(
                smote=SMOTE(random_state=self.config.SEED, k_neighbors=k_neighbors),
                random_state=self.config.SEED
            )
        elif self.config.RESAMPLING_STRATEGY == 'BorderlineSMOTE':
            resampler = BorderlineSMOTE(random_state=self.config.SEED, k_neighbors=k_neighbors)
        else:
            raise ValueError(f"Unknown resampling strategy: {self.config.RESAMPLING_STRATEGY}")
        
        X_train_resampled, y_train_resampled = resampler.fit_resample(X_train, y_train)
        
        logger.info(f"Resampled train set: {X_train_resampled.shape[0]} samples")
        logger.info("Resampled distribution:")
        for label, count in Counter(y_train_resampled).items():
            logger.info(f"  Class {label}: {count} ({count/len(y_train_resampled)*100:.1f}%)")
        
        # Feature scaling
        logger.info(f"\nApplying {self.config.SCALER_TYPE} scaling...")
        if self.config.SCALER_TYPE == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_resampled,
            'y_val': y_val,
            'y_test': y_test,
            'X_train_original': X_train,
            'y_train_original': y_train
        }
    
    def train_baseline_models(self, data_splits: Dict) -> Dict[str, Any]:
        """Train multiple baseline models for comparison"""
        logger.info("\nTraining baseline models...")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.config.SEED,
                class_weight='balanced',
                max_iter=2000,
                solver='saga',
                n_jobs=self.config.N_JOBS
            ),
            'Random Forest': RandomForestClassifier(
                random_state=self.config.SEED,
                class_weight='balanced',
                n_estimators=200,
                max_depth=15,
                n_jobs=self.config.N_JOBS
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.config.SEED,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5
            ),
            'XGBoost': XGBClassifier(
                random_state=self.config.SEED,
                n_estimators=200,
                learning_rate=0.1,
                early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                eval_metric='mlogloss',
                n_jobs=self.config.N_JOBS
            ),
            'LightGBM': LGBMClassifier(
                random_state=self.config.SEED,
                n_estimators=200,
                learning_rate=0.1,
                class_weight='balanced',
                n_jobs=self.config.N_JOBS,
                verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                random_state=self.config.SEED,
                iterations=200,
                learning_rate=0.1,
                auto_class_weights='Balanced',
                early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
                verbose=0,
                thread_count=self.config.N_JOBS
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"\nTraining {name}...")
            start_time = time.time()
            
            try:
                if name == 'CatBoost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                elif name in ['XGBoost', 'LightGBM']:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                else:
                    model.fit(X_train, y_train)
                
                train_time = time.time() - start_time
                
                # Evaluate
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)
                
                metrics = self._compute_metrics(y_val, y_pred, y_pred_proba, name)
                metrics_dict = metrics.to_dict()
                metrics_dict['training_time'] = train_time
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'training_time': train_time
                }
                
                logger.info(f"  AUC: {metrics.auc_macro:.4f}")
                logger.info(f"  F1-macro: {metrics.f1_macro:.4f}")
                logger.info(f"  MCC: {metrics.mcc:.4f}")
                logger.info(f"  Training time: {train_time:.2f}s")
                
                self.models[name] = model
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # Select best model based on AUC
        best_name = max(results.keys(), key=lambda k: results[k]['metrics'].auc_macro)
        self.best_model = results[best_name]['model']
        
        logger.info(f"\nBest baseline model: {best_name}")
        logger.info(f"AUC: {results[best_name]['metrics'].auc_macro:.4f}")
        
        return results
    
    def optimize_hyperparameters(self, data_splits: Dict, model_name: str) -> Any:
        """
        Optimize hyperparameters using Optuna.
        """
        logger.info(f"\nOptimizing {model_name} hyperparameters...")
        logger.info(f"Trials: {self.config.N_OPTUNA_TRIALS}")
        
        X_train = data_splits['X_train']
        y_train = data_splits['y_train']
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        
        def objective(trial):
            if model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 12),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
                    'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'auto_class_weights': 'Balanced',
                    'random_state': self.config.SEED,
                    'verbose': 0,
                    'thread_count': self.config.N_JOBS
                }
                model = CatBoostClassifier(**params)
                model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'random_state': self.config.SEED,
                    'n_jobs': self.config.N_JOBS
                }
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
                    'class_weight': 'balanced',
                    'random_state': self.config.SEED,
                    'n_jobs': self.config.N_JOBS,
                    'verbose': -1
                }
                model = LGBMClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
                
            else:
                raise ValueError(f"Optimization not implemented for {model_name}")
            
            y_pred_proba = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr', average='macro')
            return auc
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.config.SEED)
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config.N_OPTUNA_TRIALS,
            timeout=self.config.OPTUNA_TIMEOUT,
            show_progress_bar=True
        )
        
        logger.info(f"\nOptimization complete!")
        logger.info(f"Best AUC: {study.best_value:.4f}")
        logger.info(f"Best parameters:")
        for param, value in study.best_params.items():
            logger.info(f"  {param}: {value}")
        
        # Train final model with best parameters
        best_params = study.best_params
        best_params['random_state'] = self.config.SEED
        best_params['verbose'] = 0
        
        if model_name == 'CatBoost':
            best_params['auto_class_weights'] = 'Balanced'
            best_params['thread_count'] = self.config.N_JOBS
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
        elif model_name == 'XGBoost':
            best_params['n_jobs'] = self.config.N_JOBS
            final_model = XGBClassifier(**best_params)
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        elif model_name == 'LightGBM':
            best_params['class_weight'] = 'balanced'
            best_params['n_jobs'] = self.config.N_JOBS
            best_params['verbose'] = -1
            final_model = LGBMClassifier(**best_params)
            final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
        
        self.best_model = final_model
        
        # Save optimization results
        opt_results = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials)
        }
        
        opt_path = self.config.RESULTS_DIR / f'{model_name}_optimization.json'
        with open(opt_path, 'w') as f:
            json.dump(opt_results, f, indent=2)
        
        return final_model
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> CrossValidationResults:
        """
        Perform stratified k-fold cross-validation.
        """
        logger.info(f"\nPerforming {self.config.CV_FOLDS}-fold cross-validation...")
        
        cv = StratifiedKFold(
            n_splits=self.config.CV_FOLDS,
            shuffle=True,
            random_state=self.config.SEED
        )
        
        scoring = {
            'auc_macro': 'roc_auc_ovr',
            'auc_weighted': 'roc_auc_ovr_weighted',
            'accuracy': 'accuracy',
            'balanced_accuracy': 'balanced_accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted',
            'precision_macro': 'precision_macro',
            'recall_macro': 'recall_macro'
        }
        
        cv_results_dict = cross_validate(
            self.best_model, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=self.config.N_JOBS,
            return_train_score=True
        )
        
        # Compute mean and std
        mean_scores = {metric: np.mean(cv_results_dict[f'test_{metric}']) 
                      for metric in scoring.keys()}
        std_scores = {metric: np.std(cv_results_dict[f'test_{metric}']) 
                     for metric in scoring.keys()}
        
        logger.info("\nCross-validation results:")
        for metric in scoring.keys():
            test_mean = mean_scores[metric]
            test_std = std_scores[metric]
            train_mean = np.mean(cv_results_dict[f'train_{metric}'])
            logger.info(f"  {metric}:")
            logger.info(f"    Test:  {test_mean:.4f} ± {test_std:.4f}")
            logger.info(f"    Train: {train_mean:.4f}")
        
        cv_results = CrossValidationResults(
            model_name=type(self.best_model).__name__,
            cv_scores={k.replace('test_', ''): v for k, v in cv_results_dict.items() if k.startswith('test_')},
            mean_scores=mean_scores,
            std_scores=std_scores,
            fold_predictions=[]
        )
        
        # Save CV results
        cv_path = self.config.RESULTS_DIR / 'cross_validation_results.json'
        with open(cv_path, 'w') as f:
            json.dump({
                'mean_scores': mean_scores,
                'std_scores': std_scores
            }, f, indent=2)
        
        return cv_results
    
    def evaluate_test_set(self, data_splits: Dict) -> PerformanceMetrics:
        """
        Final evaluation on held-out test set.
        """
        logger.info("\nEvaluating on test set...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)
        
        metrics = self._compute_metrics(y_test, y_pred, y_pred_proba, 
                                       type(self.best_model).__name__)
        
        logger.info("\nTest Set Performance:")
        logger.info(f"  AUC (macro): {metrics.auc_macro:.4f}")
        logger.info(f"  AUC (weighted): {metrics.auc_weighted:.4f}")
        logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
        logger.info(f"  Balanced Accuracy: {metrics.balanced_accuracy:.4f}")
        logger.info(f"  F1 (macro): {metrics.f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {metrics.f1_weighted:.4f}")
        logger.info(f"  MCC: {metrics.mcc:.4f}")
        logger.info(f"  Cohen's Kappa: {metrics.cohen_kappa:.4f}")
        
        logger.info("\nPer-class metrics:")
        for class_id, class_metrics in metrics.per_class_metrics.items():
            logger.info(f"  Class {class_id}:")
            logger.info(f"    Precision: {class_metrics['precision']:.4f}")
            logger.info(f"    Recall: {class_metrics['recall']:.4f}")
            logger.info(f"    F1-score: {class_metrics['f1-score']:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(metrics.confusion_matrix)
        
        # Save metrics
        metrics.save(self.config.RESULTS_DIR / 'test_metrics.json')
        
        return metrics
    
    def _compute_metrics(self, y_true, y_pred, y_pred_proba, model_name) -> PerformanceMetrics:
        """Compute comprehensive performance metrics"""
        
        auc_macro = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
        auc_weighted = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Per-class metrics
        per_class = {}
        for key in report.keys():
            if key.isdigit() or key in ['0', '1', '2']:
                per_class[key] = report[key]
        
        metrics = PerformanceMetrics(
            model_name=model_name,
            auc_macro=float(auc_macro),
            auc_weighted=float(auc_weighted),
            accuracy=float(accuracy_score(y_true, y_pred)),
            balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
            f1_macro=float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            f1_micro=float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            f1_weighted=float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            precision_macro=float(precision_recall_curve(y_true == 1, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba)[0].mean()),
            recall_macro=float(precision_recall_curve(y_true == 1, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba)[1].mean()),
            mcc=float(matthews_corrcoef(y_true, y_pred)),
            cohen_kappa=float(cohen_kappa_score(y_true, y_pred)),
            confusion_matrix=confusion_matrix(y_true, y_pred),
            classification_report=report,
            per_class_metrics=per_class
        )
        
        return metrics

# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

def run_complete_pipeline(config: ExperimentConfig = None) -> Dict[str, Any]:
    """
    Execute complete training pipeline from data loading to model evaluation.
    
    This is the main entry point for training pharmacogenomic prediction models.
    
    Returns:
        Dictionary containing all results, metrics, and trained artifacts
    """
    
    # Initialize configuration
    if config is None:
        config = ExperimentConfig()
    
    logger.info("="*80)
    logger.info("STARTING PHARMACOGENOMICS PREDICTION PIPELINE")
    logger.info("="*80)
    
    results = {}
    
    try:
        # Step 1: Load Data
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("="*80)
        
        data_loader = PharmGKBDataLoader(config)
        genes_df, drugs_df, rels_df = data_loader.load_data()
        data = data_loader.create_interaction_dataset()
        
        results['data_stats'] = {
            'n_samples': len(data),
            'n_genes': data['gene_id'].nunique(),
            'n_drugs': data['drug_id'].nunique(),
            'label_distribution': data['label'].value_counts().to_dict()
        }
        
        # Step 2: Feature Engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80)
        
        feature_engineer = MultiModalFeatureEngineer(config)
        X, metadata = feature_engineer.create_features(data)
        y = data['label'].values
        
        results['feature_info'] = metadata['feature_info']
        
        # Step 3: Model Training
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*80)
        
        trainer = ModelTrainer(config)
        
        # Prepare data splits
        data_splits = trainer.prepare_data_splits(X, y)
        
        # Train baseline models
        baseline_results = trainer.train_baseline_models(data_splits)
        
        results['baseline_models'] = {
            name: {
                'auc_macro': float(res['metrics'].auc_macro),
                'f1_macro': float(res['metrics'].f1_macro),
                'accuracy': float(res['metrics'].accuracy),
                'training_time': float(res['training_time'])
            }
            for name, res in baseline_results.items()
        }
        
        # Step 4: Hyperparameter Optimization
        logger.info("\n" + "="*80)
        logger.info("STEP 4: HYPERPARAMETER OPTIMIZATION")
        logger.info("="*80)
        
        best_baseline_name = max(
            baseline_results.keys(),
            key=lambda k: baseline_results[k]['metrics'].auc_macro
        )
        
        optimized_model = trainer.optimize_hyperparameters(
            data_splits,
            best_baseline_name
        )
        
        # Step 5: Cross-Validation
        logger.info("\n" + "="*80)
        logger.info("STEP 5: CROSS-VALIDATION")
        logger.info("="*80)
        
        cv_results = trainer.cross_validate(X, y)
        
        results['cross_validation'] = {
            'mean_auc': float(cv_results.mean_scores['auc_macro']),
            'std_auc': float(cv_results.std_scores['auc_macro']),
            'mean_f1': float(cv_results.mean_scores['f1_macro']),
            'std_f1': float(cv_results.std_scores['f1_macro']),
            'mean_accuracy': float(cv_results.mean_scores['accuracy']),
            'std_accuracy': float(cv_results.std_scores['accuracy'])
        }
        
        # Step 6: Final Evaluation
        logger.info("\n" + "="*80)
        logger.info("STEP 6: FINAL EVALUATION")
        logger.info("="*80)
        
        test_metrics = trainer.evaluate_test_set(data_splits)
        
        results['test_metrics'] = test_metrics.to_dict()
        
        # Step 7: Save Model Artifacts
        logger.info("\n" + "="*80)
        logger.info("STEP 7: SAVING ARTIFACTS")
        logger.info("="*80)
        
        # Save model
        model_path = config.MODEL_DIR / 'best_model.pkl'
        joblib.dump(trainer.best_model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        # Save scaler
        scaler_path = config.MODEL_DIR / 'scaler.pkl'
        joblib.dump(trainer.scaler, scaler_path)
        logger.info(f"Scaler saved to: {scaler_path}")
        
        # Save feature engineer
        feature_eng_path = config.MODEL_DIR / 'feature_engineer.pkl'
        joblib.dump(feature_engineer, feature_eng_path)
        logger.info(f"Feature engineer saved to: {feature_eng_path}")
        
        # Save complete metadata
        metadata_combined = {
            'config': {
                'seed': config.SEED,
                'embedding_model': config.EMBEDDING_MODEL,
                'test_size': config.TEST_SIZE,
                'val_size': config.VAL_SIZE,
                'cv_folds': config.CV_FOLDS,
                'resampling_strategy': config.RESAMPLING_STRATEGY
            },
            'data_stats': results['data_stats'],
            'feature_info': results['feature_info'],
            'baseline_models': results['baseline_models'],
            'cross_validation': results['cross_validation'],
            'test_metrics': results['test_metrics']
        }
        
        metadata_path = config.MODEL_DIR / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata_combined, f, indent=2, default=str)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save complete results
        results_path = config.RESULTS_DIR / 'complete_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"\nFinal Test Performance:")
        logger.info(f"  AUC-ROC (macro): {test_metrics.auc_macro:.4f}")
        logger.info(f"  F1-Score (macro): {test_metrics.f1_macro:.4f}")
        logger.info(f"  Accuracy: {test_metrics.accuracy:.4f}")
        logger.info(f"  MCC: {test_metrics.mcc:.4f}")
        logger.info(f"\nAll artifacts saved to: {config.OUTPUT_DIR}")
        
        results['success'] = True
        results['output_dir'] = str(config.OUTPUT_DIR)
        results['model_path'] = str(model_path)
        results['scaler_path'] = str(scaler_path)
        
        return results
        
    except Exception as e:
        logger.error(f"\n{'='*80}")
        logger.error("PIPELINE FAILED")
        logger.error(f"{'='*80}")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        results['success'] = False
        results['error'] = str(e)
        
        return results

# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

def main():
    """Main entry point for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pharmacogenomics Interaction Prediction Framework'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing PharmGKB TSV files'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-mpnet-base-v2',
        help='Sentence transformer model name'
    )
    
    parser.add_argument(
        '--no-sequences',
        action='store_true',
        help='Disable gene sequence fetching'
    )
    
    parser.add_argument(
        '--no-pubmed',
        action='store_true',
        help='Disable PubMed abstract fetching'
    )
    
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=10,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--n-trials',
        type=int,
        default=100,
        help='Number of Optuna optimization trials'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = ExperimentConfig()
    
    # Update from arguments
    config.SEED = args.seed
    config.EMBEDDING_MODEL = args.embedding_model
    config.FETCH_GENE_SEQUENCES = not args.no_sequences
    config.FETCH_PUBMED_ABSTRACTS = not args.no_pubmed
    config.CV_FOLDS = args.cv_folds
    config.N_OPTUNA_TRIALS = args.n_trials
    
    if args.data_dir:
        config.DATA_DIR = Path(args.data_dir)
        config.GENES_TSV = config.DATA_DIR / "genes.tsv"
        config.DRUGS_TSV = config.DATA_DIR / "drugs.tsv"
        config.RELS_TSV = config.DATA_DIR / "relationships.tsv"
    
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
        config.MODEL_DIR = config.OUTPUT_DIR / "models"
        config.RESULTS_DIR = config.OUTPUT_DIR / "results"
        config.FIGURES_DIR = config.OUTPUT_DIR / "figures"
    
    # Run pipeline
    results = run_complete_pipeline(config)
    
    if results['success']:
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"\nModel artifacts saved to: {results['output_dir']}")
        print(f"\nTest Performance:")
        print(f"  AUC: {results['test_metrics']['auc_macro']:.4f}")
        print(f"  F1:  {results['test_metrics']['f1_macro']:.4f}")
        print(f"  ACC: {results['test_metrics']['accuracy']:.4f}")
        return 0
    else:
        print("\n" + "="*80)
        print("FAILED!")
        print("="*80)
        print(f"\nError: {results.get('error', 'Unknown error')}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

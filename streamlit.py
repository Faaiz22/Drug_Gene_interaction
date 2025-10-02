"""
Complete Training Pipeline with Streamlit Interface
Integrates with gene_drug_framework.txt for full model training

This script provides an interactive interface for:
1. Configuring experiments
2. Loading and preprocessing data
3. Engineering multi-modal features
4. Training and optimizing models
5. Evaluating performance
6. Exporting trained models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import time
import warnings
from datetime import datetime
import sys
import traceback

warnings.filterwarnings('ignore')

# Import framework components
try:
    from gene_drug_framework import (
        ExperimentConfig,
        PharmGKBDataLoader,
        MultiModalFeatureEngineer,
        ModelTrainer,
        PerformanceMetrics,
        CrossValidationResults
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    st.error("""
    ⚠️ **Framework not found!**
    
    Please ensure `gene_drug_framework.py` is in the same directory.
    You can extract it from the provided `gene_drug_framework.txt` file.
    """)
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Training Pipeline - Pharmacogenomics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'config': None,
        'data_loaded': False,
        'features_created': False,
        'model_trained': False,
        'results': {},
        'data': None,
        'X': None,
        'y': None,
        'metadata': None
    }

class InteractiveTrainingPipeline:
    """Interactive training pipeline with progress tracking"""
    
    def __init__(self):
        self.config = None
        self.data_loader = None
        self.feature_engineer = None
        self.trainer = None
        self.results = {}
    
    def create_config(self, params: dict) -> ExperimentConfig:
        """Create configuration from user parameters"""
        config = ExperimentConfig()
        
        # Update with user parameters
        config.SEED = params['seed']
        config.EMBEDDING_MODEL = params['embedding_model']
        config.TEST_SIZE = params['test_size']
        config.VAL_SIZE = params['val_size']
        config.CV_FOLDS = params['cv_folds']
        config.N_OPTUNA_TRIALS = params['n_trials']
        config.RESAMPLING_STRATEGY = params['resampling']
        config.USE_SEQUENCE_FEATURES = params['use_sequences']
        config.FETCH_PUBMED_ABSTRACTS = params['fetch_pubmed']
        config.USE_STATISTICAL_FEATURES = params['use_stats']
        config.SCALER_TYPE = params['scaler_type']
        
        self.config = config
        return config
    
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess PharmGKB data"""
        self.data_loader = PharmGKBDataLoader(self.config)
        genes_df, drugs_df, rels_df = self.data_loader.load_data()
        data = self.data_loader.create_interaction_dataset()
        
        self.results['dataset_size'] = len(data)
        self.results['n_genes'] = data['gene_id'].nunique()
        self.results['n_drugs'] = data['drug_id'].nunique()
        self.results['label_distribution'] = data['label'].value_counts().to_dict()
        
        return data
    
    def create_features(self, data: pd.DataFrame) -> tuple:
        """Create multi-modal feature matrix"""
        self.feature_engineer = MultiModalFeatureEngineer(self.config)
        X, metadata = self.feature_engineer.create_features(data)
        y = data['label'].values
        
        self.results['feature_shape'] = X.shape
        self.results['feature_info'] = metadata['feature_info']
        
        return X, y, metadata
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train and optimize models"""
        self.trainer = ModelTrainer(self.config)
        
        # Prepare data splits
        data_splits = self.trainer.prepare_data_splits(X, y)
        self.results['split_info'] = {
            'train_size': len(data_splits['y_train']),
            'val_size': len(data_splits['y_val']),
            'test_size': len(data_splits['y_test'])
        }
        
        # Train baseline models
        baseline_results = self.trainer.train_baseline_models(data_splits)
        self.results['baseline_results'] = {
            name: {
                'auc_macro': float(res['metrics'].auc_macro),
                'f1_macro': float(res['metrics'].f1_macro),
                'accuracy': float(res['metrics'].accuracy),
                'training_time': float(res['training_time'])
            }
            for name, res in baseline_results.items()
        }
        
        # Get best model name
        best_model_name = max(
            baseline_results.keys(),
            key=lambda k: baseline_results[k]['metrics'].auc_macro
        )
        self.results['best_baseline'] = best_model_name
        
        # Optimize hyperparameters
        optimized_model = self.trainer.optimize_hyperparameters(
            data_splits,
            best_model_name
        )
        
        # Cross-validation
        cv_results = self.trainer.cross_validate(X, y)
        self.results['cv_results'] = {
            'mean_auc': float(cv_results.mean_scores['auc_macro']),
            'std_auc': float(cv_results.std_scores['auc_macro']),
            'mean_f1': float(cv_results.mean_scores['f1_macro']),
            'std_f1': float(cv_results.std_scores['f1_macro']),
            'mean_accuracy': float(cv_results.mean_scores['accuracy']),
            'std_accuracy': float(cv_results.std_scores['accuracy'])
        }
        
        # Final evaluation
        test_metrics = self.trainer.evaluate_test_set(data_splits)
        self.results['test_metrics'] = {
            'auc_macro': float(test_metrics.auc_macro),
            'auc_weighted': float(test_metrics.auc_weighted),
            'accuracy': float(test_metrics.accuracy),
            'balanced_accuracy': float(test_metrics.balanced_accuracy),
            'f1_macro': float(test_metrics.f1_macro),
            'f1_weighted': float(test_metrics.f1_weighted),
            'mcc': float(test_metrics.mcc),
            'cohen_kappa': float(test_metrics.cohen_kappa),
            'confusion_matrix': test_metrics.confusion_matrix.tolist(),
            'classification_report': test_metrics.classification_report
        }
        
        return self.results
    
    def save_artifacts(self) -> dict:
        """Save trained model and artifacts"""
        import joblib
        
        # Save model
        model_path = self.config.MODEL_DIR / 'best_model.pkl'
        joblib.dump(self.trainer.best_model, model_path)
        
        # Save scaler
        scaler_path = self.config.MODEL_DIR / 'scaler.pkl'
        joblib.dump(self.trainer.scaler, scaler_path)
        
        # Save results
        results_path = self.config.RESULTS_DIR / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        paths = {
            'model': str(model_path),
            'scaler': str(scaler_path),
            'results': str(results_path),
            'output_dir': str(self.config.OUTPUT_DIR)
        }
        
        return paths

def main():
    # Title
    st.markdown('<h1 class="big-title">⚙️ Training Pipeline</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <p style="font-size:1.1rem; margin:0;">
    <strong>Complete Training Pipeline</strong> for pharmacogenomics interaction prediction models.
    Configure parameters, monitor training progress, and evaluate results in real-time.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Experiment settings
        with st.expander("📋 Experiment Settings", expanded=True):
            seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999)
            experiment_name = st.text_input(
                "Experiment Name",
                value=f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Model settings
        with st.expander("🤖 Model Configuration", expanded=True):
            embedding_model = st.selectbox(
                "Embedding Model",
                [
                    "all-mpnet-base-v2",
                    "all-MiniLM-L6-v2",
                    "paraphrase-MiniLM-L6-v2"
                ],
                help="Transformer model for semantic embeddings"
            )
            
            resampling = st.selectbox(
                "Resampling Strategy",
                ["SMOTE-Tomek", "SMOTE", "ADASYN", "BorderlineSMOTE"],
                help="Strategy to handle class imbalance"
            )
            
            scaler_type = st.selectbox(
                "Feature Scaler",
                ["robust", "standard"],
                help="Method for feature normalization"
            )
        
        # Data split settings
        with st.expander("📊 Data Split", expanded=False):
            test_size = st.slider("Test Size", 0.1, 0.3, 0.15, 0.05)
            val_size = st.slider("Validation Size", 0.1, 0.3, 0.15, 0.05)
        
        # Training settings
        with st.expander("🏋️ Training Configuration", expanded=False):
            cv_folds = st.number_input("CV Folds", value=10, min_value=3, max_value=20)
            n_trials = st.number_input(
                "Optuna Trials",
                value=100,
                min_value=10,
                max_value=500,
                help="Number of hyperparameter optimization trials"
            )
        
        # Feature settings
        with st.expander("🧬 Feature Engineering", expanded=False):
            use_sequences = st.checkbox("Fetch Gene Sequences", value=True)
            fetch_pubmed = st.checkbox("Fetch PubMed Abstracts", value=False)
            use_stats = st.checkbox("Use Statistical Features", value=True)
        
        st.markdown("---")
        
        # Create configuration button
        if st.button("💾 Initialize Configuration", use_container_width=True):
            params = {
                'seed': seed,
                'embedding_model': embedding_model,
                'test_size': test_size,
                'val_size': val_size,
                'cv_folds': cv_folds,
                'n_trials': n_trials,
                'resampling': resampling,
                'use_sequences': use_sequences,
                'fetch_pubmed': fetch_pubmed,
                'use_stats': use_stats,
                'scaler_type': scaler_type
            }
            
            pipeline = InteractiveTrainingPipeline()
            config = pipeline.create_config(params)
            
            st.session_state.pipeline_state['config'] = config
            st.session_state.pipeline_state['pipeline'] = pipeline
            
            st.success("Configuration initialized!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Run Training", "📈 Results", "🔍 Model Analysis", "📦 Export"
    ])
    
    # Tab 1: Run Training
    with tab1:
        st.markdown('<div class="section-header">Training Pipeline</div>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline_state['config']:
            st.warning("⚠️ Please initialize configuration in the sidebar first")
            st.stop()
        
        config = st.session_state.pipeline_state['config']
        
        # Display configuration summary
        st.subheader("Configuration Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            **Experiment Settings**
            - Seed: {config.SEED}
            - Embedding: {config.EMBEDDING_MODEL}
            """)
        
        with col2:
            st.markdown(f"""
            **Data Split**
            - Test: {config.TEST_SIZE*100:.0f}%
            - Validation: {config.VAL_SIZE*100:.0f}%
            - CV Folds: {config.CV_FOLDS}
            """)
        
        with col3:
            st.markdown(f"""
            **Features**
            - Sequences: {'✓' if config.USE_SEQUENCE_FEATURES else '✗'}
            - PubMed: {'✓' if config.FETCH_PUBMED_ABSTRACTS else '✗'}
            - Statistics: {'✓' if config.USE_STATISTICAL_FEATURES else '✗'}
            """)
        
        st.markdown("---")
        
        # Training steps
        st.subheader("Training Steps")
        
        # Step 1: Load Data
        if st.button("1️⃣ Load Data", use_container_width=True, disabled=st.session_state.pipeline_state['data_loaded']):
            with st.spinner("Loading PharmGKB data..."):
                try:
                    pipeline = st.session_state.pipeline_state['pipeline']
                    data = pipeline.load_data()
                    
                    st.session_state.pipeline_state['data'] = data
                    st.session_state.pipeline_state['data_loaded'] = True
                    
                    st.success("✅ Data loaded successfully!")
                    
                    # Display data summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Interactions", len(data))
                    with col2:
                        st.metric("Unique Genes", data['gene_id'].nunique())
                    with col3:
                        st.metric("Unique Drugs", data['drug_id'].nunique())
                    
                    # Label distribution
                    st.subheader("Label Distribution")
                    label_counts = data['label'].value_counts().sort_index()
                    label_names = {0: 'Not Associated', 1: 'Associated', 2: 'Ambiguous'}
                    
                    fig = px.bar(
                        x=[label_names[i] for i in label_counts.index],
                        y=label_counts.values,
                        labels={'x': 'Class', 'y': 'Count'},
                        title='Class Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
                    st.code(traceback.format_exc())
        
        if st.session_state.pipeline_state['data_loaded']:
            st.info(f"✓ Data loaded: {len(st.session_state.pipeline_state['data'])} interactions")
        
        # Step 2: Create Features
        if st.button("2️⃣ Create Features", use_container_width=True, 
                    disabled=not st.session_state.pipeline_state['data_loaded'] or 
                    st.session_state.pipeline_state['features_created']):
            with st.spinner("Engineering multi-modal features... This may take several minutes."):
                try:
                    pipeline = st.session_state.pipeline_state['pipeline']
                    data = st.session_state.pipeline_state['data']
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Creating embeddings...")
                    progress_bar.progress(20)
                    
                    X, y, metadata = pipeline.create_features(data)
                    
                    status_text.text("Features created successfully!")
                    progress_bar.progress(100)
                    
                    st.session_state.pipeline_state['X'] = X
                    st.session_state.pipeline_state['y'] = y
                    st.session_state.pipeline_state['metadata'] = metadata
                    st.session_state.pipeline_state['features_created'] = True
                    
                    st.success("✅ Features created successfully!")
                    
                    # Display feature info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Feature Dimension", X.shape[1])
                        st.metric("Samples", X.shape[0])
                    with col2:
                        st.metric("Embedding Dim", metadata['feature_info']['gene_embedding_dim'])
                        st.metric("Interaction Features", metadata['feature_info']['interaction_dims'])
                    
                except Exception as e:
                    st.error(f"Error creating features: {str(e)}")
                    st.code(traceback.format_exc())
        
        if st.session_state.pipeline_state['features_created']:
            X = st.session_state.pipeline_state['X']
            st.info(f"✓ Features created: {X.shape}")
        
        # Step 3: Train Models
        if st.button("3️⃣ Train Models", use_container_width=True, type="primary",
                    disabled=not st.session_state.pipeline_state['features_created'] or 
                    st.session_state.pipeline_state['model_trained']):
            with st.spinner("Training models... This may take 30+ minutes."):
                try:
                    pipeline = st.session_state.pipeline_state['pipeline']
                    X = st.session_state.pipeline_state['X']
                    y = st.session_state.pipeline_state['y']
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Preparing data splits...")
                    progress_bar.progress(10)
                    
                    status_text.text("Training baseline models...")
                    progress_bar.progress(30)
                    
                    status_text.text("Optimizing hyperparameters...")
                    progress_bar.progress(60)
                    
                    results = pipeline.train_models(X, y)
                    
                    status_text.text("Training complete!")
                    progress_bar.progress(100)
                    
                    st.session_state.pipeline_state['results'] = results
                    st.session_state.pipeline_state['model_trained'] = True
                    
                    st.success("✅ Models trained successfully!")
                    
                    # Display quick results
                    st.subheader("Quick Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Test AUC", f"{results['test_metrics']['auc_macro']:.4f}")
                    with col2:
                        st.metric("Test F1", f"{results['test_metrics']['f1_macro']:.4f}")
                    with col3:
                        st.metric("Test Accuracy", f"{results['test_metrics']['accuracy']:.4f}")
                    
                except Exception as e:
                    st.error(f"Error training models: {str(e)}")
                    st.code(traceback.format_exc())
        
        if st.session_state.pipeline_state['model_trained']:
            results = st.session_state.pipeline_state['results']
            st.info(f"✓ Model trained - Best: {results.get('best_baseline', 'N/A')}")
    
    # Tab 2: Results
    with tab2:
        st.markdown('<div class="section-header">Training Results</div>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline_state['model_trained']:
            st.warning("Complete training first to view results")
            st.stop()
        
        results = st.session_state.pipeline_state['results']
        
        # Baseline model comparison
        st.subheader("Baseline Model Comparison")
        
        baseline_df = pd.DataFrame(results['baseline_results']).T
        baseline_df = baseline_df.sort_values('auc_macro', ascending=False)
        
        fig = px.bar(
            baseline_df.reset_index(),
            x='index',
            y=['auc_macro', 'f1_macro', 'accuracy'],
            barmode='group',
            labels={'index': 'Model', 'value': 'Score', 'variable': 'Metric'},
            title='Baseline Model Performance'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(baseline_df, use_container_width=True)
        
        # Cross-validation results
        st.subheader("Cross-Validation Results")
        
        cv_res = results['cv_results']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Mean AUC",
                f"{cv_res['mean_auc']:.4f}",
                delta=f"±{cv_res['std_auc']:.4f}"
            )
        
        with col2:
            st.metric(
                "Mean F1-Score",
                f"{cv_res['mean_f1']:.4f}",
                delta=f"±{cv_res['std_f1']:.4f}"
            )
        
        with col3:
            st.metric(
                "Mean Accuracy",
                f"{cv_res['mean_accuracy']:.4f}",
                delta=f"±{cv_res['std_accuracy']:.4f}"
            )
        
        # Test set performance
        st.subheader("Test Set Performance")
        
        test_metrics = results['test_metrics']
        
        # Metrics grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("AUC-ROC", f"{test_metrics['auc_macro']:.4f}")
            st.metric("AUC Weighted", f"{test_metrics['auc_weighted']:.4f}")
        
        with col2:
            st.metric("Accuracy", f"{test_metrics['accuracy']:.4f}")
            st.metric("Balanced Acc.", f"{test_metrics['balanced_accuracy']:.4f}")
        
        with col3:
            st.metric("F1-Macro", f"{test_metrics['f1_macro']:.4f}")
            st.metric("F1-Weighted", f"{test_metrics['f1_weighted']:.4f}")
        
        with col4:
            st.metric("MCC", f"{test_metrics['mcc']:.4f}")
            st.metric("Cohen's Kappa", f"{test_metrics['cohen_kappa']:.4f}")
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        
        cm = np.array(test_metrics['confusion_matrix'])
        
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Not Associated', 'Associated', 'Ambiguous'],
            y=['Not Associated', 'Associated', 'Ambiguous'],
            text_auto=True,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.subheader("Classification Report")
        
        if 'classification_report' in test_metrics:
            report_df = pd.DataFrame(test_metrics['classification_report']).transpose()
            st.dataframe(report_df, use_container_width=True)
    
    # Tab 3: Model Analysis
    with tab3:
        st.markdown('<div class="section-header">Model Analysis</div>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline_state['model_trained']:
            st.warning("Complete training first to view analysis")
            st.stop()
        
        results = st.session_state.pipeline_state['results']
        
        # Dataset statistics
        st.subheader("Dataset Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", results['dataset_size'])
            st.metric("Unique Genes", results['n_genes'])
        
        with col2:
            st.metric("Unique Drugs", results['n_drugs'])
            st.metric("Feature Dimension", results['feature_shape'][1])
        
        with col3:
            split_info = results['split_info']
            st.metric("Train Size", split_info['train_size'])
            st.metric("Test Size", split_info['test_size'])
        
        # Label distribution
        st.subheader("Label Distribution")
        
        label_dist = results['label_distribution']
        label_names = {0: 'Not Associated', 1: 'Associated', 2: 'Ambiguous'}
        
        dist_df = pd.DataFrame([
            {'Class': label_names.get(int(k), k), 'Count': v}
            for k, v in label_dist.items()
        ])
        
        fig = px.pie(
            dist_df,
            values='Count',
            names='Class',
            title='Class Distribution',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature information
        st.subheader("Feature Composition")
        
        feat_info = results['feature_info']
        
        feature_breakdown = {
            'Gene Embeddings': feat_info['gene_embedding_dim'],
            'Drug Embeddings': feat_info['drug_embedding_dim'],
            'Text Embeddings': feat_info['text_embedding_dim'],
            'Interaction Features': feat_info['interaction_dims'],
            'Sequence Features': feat_info['sequence_features'],
            'Statistical Features': feat_info['statistical_features']
        }
        
        feat_df = pd.DataFrame([
            {'Component': k, 'Dimensions': v}
            for k, v in feature_breakdown.items()
            if v > 0
        ])
        
        fig = px.bar(
            feat_df,
            x='Component',
            y='Dimensions',
            title='Feature Components',
            color='Dimensions',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model training time
        st.subheader("Training Time Comparison")
        
        if 'baseline_results' in results:
            time_df = pd.DataFrame([
                {'Model': k, 'Time (seconds)': v['training_time']}
                for k, v in results['baseline_results'].items()
            ]).sort_values('Time (seconds)')
            
            fig = px.bar(
                time_df,
                x='Model',
                y='Time (seconds)',
                title='Model Training Time',
                color='Time (seconds)',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Export
    with tab4:
        st.markdown('<div class="section-header">Export Model & Results</div>', unsafe_allow_html=True)
        
        if not st.session_state.pipeline_state['model_trained']:
            st.warning("Complete training first to export artifacts")
            st.stop()
        
        st.info("Save trained model and artifacts for deployment")
        
        if st.button("💾 Save Model Artifacts", use_container_width=True, type="primary"):
            with st.spinner("Saving artifacts..."):
                try:
                    pipeline = st.session_state.pipeline_state['pipeline']
                    paths = pipeline.save_artifacts()
                    
                    st.success("✅ Artifacts saved successfully!")
                    
                    st.markdown("### Saved Files")
                    
                    st.markdown(f"""
                    - **Model**: `{paths['model']}`
                    - **Scaler**: `{paths['scaler']}`
                    - **Results**: `{paths['results']}`
                    - **Output Directory**: `{paths['output_dir']}`
                    """)
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("""
                    ### Next Steps
                    
                    1. **Test the model**: Use `app.py` to make predictions
                    2. **Review results**: Check the output directory for detailed metrics
                    3. **Deploy**: Copy model files to production environment
                    4. **Document**: Update documentation with model performance
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error saving artifacts: {str(e)}")
                    st.code(traceback.format_exc())
        
        # Download results
        st.subheader("Download Results")
        
        results = st.session_state.pipeline_state['results']
        
        results_json = json.dumps(results, indent=2, default=str)
        
        st.download_button(
            "📥 Download Results (JSON)",
            results_json,
            f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "application/json",
            use_container_width=True
        )
        
        # Model card
        st.subheader("Model Card")
        
        st.markdown(f"""
        ### Model Information
        
        **Model Type**: {results.get('best_baseline', 'Ensemble')}
        
        **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        **Performance Summary**:
        - AUC-ROC: {results['test_metrics']['auc_macro']:.4f}
        - F1-Score: {results['test_metrics']['f1_macro']:.4f}
        - Accuracy: {results['test_metrics']['accuracy']:.4f}
        
        **Dataset**:
        - Total Samples: {results['dataset_size']}
        - Genes: {results['n_genes']}
        - Drugs: {results['n_drugs']}
        
        **Features**:
        - Total Dimensions: {results['feature_shape'][1]}
        - Embedding Model: {results['feature_info']['embedding_model']}
        
        **Training Configuration**:
        - Resampling: {st.session_state.pipeline_state['config'].RESAMPLING_STRATEGY}
        - CV Folds: {st.session_state.pipeline_state['config'].CV_FOLDS}
        - Optimization Trials: {st.session_state.pipeline_state['config'].N_OPTUNA_TRIALS}
        
        **Intended Use**:
        - Research and educational purposes
        - Pharmacogenomics interaction prediction
        - Clinical decision support (with validation)
        
        **Limitations**:
        - Based on PharmGKB annotations
        - May not capture rare interactions
        - Requires genetic testing confirmation
        - Not FDA approved for clinical use
        """)

if __name__ == "__main__":
    main()
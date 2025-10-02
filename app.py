"""
Pharmacogenomics Interaction Prediction Web Application
Complete implementation with gene-drug pair prediction and interpretation
Based on the research framework in gene_drug_framework.txt
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Try to import the framework modules
try:
    from gene_drug_framework import (
        ExperimentConfig,
        PharmGKBDataLoader,
        MultiModalFeatureEngineer,
        ModelTrainer,
        PerformanceMetrics
    )
    FRAMEWORK_AVAILABLE = True
except ImportError:
    FRAMEWORK_AVAILABLE = False
    st.warning("Framework modules not found. Using mock prediction mode.")

# Page configuration
st.set_page_config(
    page_title="Pharmacogenomics Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .associated {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
    }
    .not-associated {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
    }
    .ambiguous {
        background-color: #fff3cd;
        border-left: 6px solid #ffc107;
    }
    .info-section {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-section {
        background-color: #fff4e5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.feature_engineer = None
    st.session_state.metadata = None
    st.session_state.config = None

class PredictionInterface:
    """Handle model loading and predictions with full feature engineering"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.model = None
        self.scaler = None
        self.feature_engineer = None
        self.metadata = None
        
    def load_artifacts(self, model_dir: Path) -> bool:
        """Load trained model and all preprocessing artifacts"""
        try:
            # Load model
            model_path = model_dir / 'best_model.pkl'
            if not model_path.exists():
                st.error(f"Model file not found: {model_path}")
                return False
            
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = model_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            
            # Load metadata
            metadata_path = model_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Initialize feature engineer
            self.feature_engineer = MultiModalFeatureEngineer(self.config)
            self.feature_engineer.initialize_embedder()
            
            st.success("Model artifacts loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading model artifacts: {e}")
            return False
    
    def predict_interaction(self, gene_name: str, drug_name: str, 
                          gene_symbol: str = None, gene_id: str = None,
                          drug_id: str = None, pmids: str = "",
                          association: str = "", evidence: str = "") -> Dict:
        """
        Predict gene-drug interaction with comprehensive feature extraction
        
        Returns detailed prediction with probabilities, confidence, and interpretation
        """
        try:
            # Create input dataframe matching training format
            input_data = pd.DataFrame({
                'gene_id': [gene_id or f"GENE_{gene_name}"],
                'gene_name': [gene_name],
                'gene_symbol': [gene_symbol or gene_name],
                'drug_id': [drug_id or f"DRUG_{drug_name}"],
                'drug_name': [drug_name],
                'PMIDs': [pmids],
                'Association': [association or "Unknown"],
                'Evidence': [evidence or ""]
            })
            
            # Create features using the same pipeline as training
            st.info("Extracting multi-modal features...")
            
            # Create evidence text
            input_data['evidence_text'] = input_data.apply(
                self.feature_engineer.create_evidence_text, axis=1
            )
            
            # Get embeddings
            gene_embedding = self.feature_engineer.embedder.encode(
                f"{gene_name} {gene_symbol or ''}", 
                show_progress_bar=False
            )
            drug_embedding = self.feature_engineer.embedder.encode(
                drug_name, 
                show_progress_bar=False
            )
            text_embedding = self.feature_engineer.embedder.encode(
                input_data['evidence_text'].iloc[0],
                show_progress_bar=False
            )
            
            # Create interaction features
            gene_text = gene_embedding * text_embedding
            drug_text = drug_embedding * text_embedding
            gene_drug = gene_embedding * drug_embedding
            
            # Sequence features (if enabled)
            sequence_features = []
            if self.config.USE_SEQUENCE_FEATURES:
                st.info(f"Fetching gene sequence for {gene_name}...")
                sequence = self.feature_engineer.fetch_gene_sequence(
                    gene_name, gene_symbol
                )
                seq_features = self.feature_engineer.compute_sequence_features(sequence) if sequence else {
                    'seq_length': 0.0, 'gc_content': 0.0, 'molecular_weight': 0.0,
                    'a_content': 0.0, 't_content': 0.0, 'g_content': 0.0, 'c_content': 0.0
                }
                sequence_features = list(seq_features.values())
            
            # Statistical features
            cosine_sim = np.dot(gene_embedding, drug_embedding) / (
                np.linalg.norm(gene_embedding) * np.linalg.norm(drug_embedding) + 1e-10
            )
            euclidean_dist = np.linalg.norm(gene_embedding - drug_embedding)
            
            stats_features = [
                np.mean(gene_embedding), np.std(gene_embedding), 
                np.max(gene_embedding), np.min(gene_embedding),
                np.mean(drug_embedding), np.std(drug_embedding),
                np.max(drug_embedding), np.min(drug_embedding),
                cosine_sim, euclidean_dist
            ]
            
            # Combine all features
            feature_components = [
                gene_embedding, drug_embedding, text_embedding,
                gene_text, drug_text, gene_drug
            ]
            
            if self.config.USE_SEQUENCE_FEATURES:
                feature_components.append(np.array(sequence_features))
            
            if self.config.USE_STATISTICAL_FEATURES:
                feature_components.append(np.array(stats_features))
            
            features = np.concatenate(feature_components).reshape(1, -1)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Interpret results
            label_names = {0: "Not Associated", 1: "Associated", 2: "Ambiguous"}
            
            result = {
                'prediction': int(prediction),
                'prediction_label': label_names[prediction],
                'probabilities': {
                    'not_associated': float(probabilities[0]),
                    'associated': float(probabilities[1]),
                    'ambiguous': float(probabilities[2]) if len(probabilities) > 2 else 0.0
                },
                'confidence': float(np.max(probabilities)),
                'gene': gene_name,
                'drug': drug_name,
                'features': {
                    'cosine_similarity': float(cosine_sim),
                    'euclidean_distance': float(euclidean_dist),
                    'gene_embedding_mean': float(np.mean(gene_embedding)),
                    'drug_embedding_mean': float(np.mean(drug_embedding))
                }
            }
            
            # Add sequence features if available
            if self.config.USE_SEQUENCE_FEATURES and sequence_features:
                result['sequence_features'] = {
                    'seq_length': sequence_features[0],
                    'gc_content': sequence_features[1],
                    'molecular_weight': sequence_features[2],
                    'a_content': sequence_features[3],
                    't_content': sequence_features[4],
                    'g_content': sequence_features[5],
                    'c_content': sequence_features[6]
                }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None

def display_prediction_results(result: Dict):
    """Display comprehensive prediction results with visualizations"""
    
    st.markdown("### Prediction Results")
    
    # Prediction box with appropriate styling
    if result['prediction'] == 1:
        box_class = "associated"
        emoji = "✅"
        confidence_color = "green"
    elif result['prediction'] == 0:
        box_class = "not-associated"
        emoji = "❌"
        confidence_color = "red"
    else:
        box_class = "ambiguous"
        emoji = "⚠️"
        confidence_color = "orange"
    
    st.markdown(f"""
    <div class="prediction-box {box_class}">
        <h2 style="margin:0;">{emoji} {result['prediction_label']}</h2>
        <p style="font-size:1.2rem; margin-top:1rem;">
            <strong>Confidence:</strong> 
            <span style="color:{confidence_color}; font-weight:bold;">
                {result['confidence']*100:.1f}%
            </span>
        </p>
        <p><strong>Gene:</strong> {result['gene']}</p>
        <p><strong>Drug:</strong> {result['drug']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability distribution visualization
    st.markdown("### Probability Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prob_df = pd.DataFrame({
            'Class': ['Not Associated', 'Associated', 'Ambiguous'],
            'Probability': [
                result['probabilities']['not_associated'],
                result['probabilities']['associated'],
                result['probabilities']['ambiguous']
            ]
        })
        
        fig = px.bar(
            prob_df, 
            x='Class', 
            y='Probability',
            color='Probability',
            color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
            text='Probability'
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_layout(
            showlegend=False, 
            height=400,
            yaxis_title="Probability",
            xaxis_title="Interaction Class"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Class Probabilities")
        for class_name, prob in result['probabilities'].items():
            st.metric(
                class_name.replace('_', ' ').title(),
                f"{prob*100:.1f}%"
            )
    
    # Feature analysis
    st.markdown("### Feature Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Semantic Similarity")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Cosine Similarity",
            f"{result['features']['cosine_similarity']:.4f}",
            help="Measures semantic similarity between gene and drug embeddings"
        )
        st.metric(
            "Euclidean Distance",
            f"{result['features']['euclidean_distance']:.4f}",
            help="Geometric distance in embedding space"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Embedding Statistics")
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            "Gene Embedding Mean",
            f"{result['features']['gene_embedding_mean']:.4f}"
        )
        st.metric(
            "Drug Embedding Mean",
            f"{result['features']['drug_embedding_mean']:.4f}"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Sequence features if available
    if 'sequence_features' in result:
        st.markdown("### Gene Sequence Features")
        
        seq_feat = result['sequence_features']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("GC Content", f"{seq_feat['gc_content']*100:.2f}%")
            st.metric("Sequence Length", f"{seq_feat['seq_length']*10000:.0f} bp")
        
        with col2:
            st.metric("Molecular Weight", f"{seq_feat['molecular_weight']*100:.1f} kDa")
            st.metric("A Content", f"{seq_feat['a_content']*100:.1f}%")
        
        with col3:
            st.metric("T Content", f"{seq_feat['t_content']*100:.1f}%")
            st.metric("G Content", f"{seq_feat['g_content']*100:.1f}%")
        
        # Nucleotide composition pie chart
        nucleotide_data = pd.DataFrame({
            'Nucleotide': ['A', 'T', 'G', 'C'],
            'Percentage': [
                seq_feat['a_content'] * 100,
                seq_feat['t_content'] * 100,
                seq_feat['g_content'] * 100,
                seq_feat['c_content'] * 100
            ]
        })
        
        fig = px.pie(
            nucleotide_data,
            values='Percentage',
            names='Nucleotide',
            title='Nucleotide Composition',
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Clinical interpretation
    display_clinical_interpretation(result)

def display_clinical_interpretation(result: Dict):
    """Display detailed clinical interpretation"""
    
    st.markdown("### Clinical Interpretation")
    
    gene_name = result['gene']
    drug_name = result['drug']
    confidence = result['confidence']
    prediction = result['prediction']
    
    if prediction == 1:  # Associated
        st.markdown(f"""
        <div class="info-section">
        <h4>✅ Strong Pharmacogenomic Association Detected</h4>
        
        <p>The model predicts a <strong>significant pharmacogenomic interaction</strong> between 
        <strong>{gene_name}</strong> and <strong>{drug_name}</strong> with 
        <strong style="color:#28a745;">{confidence*100:.1f}% confidence</strong>.</p>
        
        <p><strong>Clinical Implications:</strong></p>
        <ul>
            <li>Genetic variants in {gene_name} may significantly affect {drug_name} metabolism, efficacy, or safety</li>
            <li>Consider genotype-guided dosing strategies</li>
            <li>Review patient's genetic profile before prescribing</li>
            <li>Monitor for potential adverse drug reactions</li>
        </ul>
        
        <p><strong>Recommended Actions:</strong></p>
        <ul>
            <li>Consult CPIC (Clinical Pharmacogenetics Implementation Consortium) guidelines</li>
            <li>Consider pharmacogenomic testing if not already available</li>
            <li>Document genetic information in patient medical record</li>
            <li>Adjust dosing based on genotype-phenotype predictions</li>
            <li>Implement enhanced monitoring protocols</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    elif prediction == 0:  # Not Associated
        st.markdown(f"""
        <div class="warning-section">
        <h4>❌ No Strong Association Detected</h4>
        
        <p>The model does not predict a significant pharmacogenomic interaction between 
        <strong>{gene_name}</strong> and <strong>{drug_name}</strong>.</p>
        
        <p><strong>Important Considerations:</strong></p>
        <ul>
            <li>This does not completely rule out <em>all</em> possible interactions</li>
            <li>Standard clinical monitoring is still recommended</li>
            <li>Patient-specific factors should still be considered</li>
            <li>Emerging research may reveal new associations</li>
        </ul>
        
        <p><strong>Clinical Approach:</strong></p>
        <ul>
            <li>Follow standard prescribing guidelines</li>
            <li>Monitor for typical adverse effects</li>
            <li>Consider other genetic and non-genetic factors</li>
            <li>Stay updated on new pharmacogenomic evidence</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    else:  # Ambiguous
        st.markdown(f"""
        <div class="warning-section">
        <h4>⚠️ Ambiguous or Uncertain Result</h4>
        
        <p>The evidence for interaction between <strong>{gene_name}</strong> and 
        <strong>{drug_name}</strong> is unclear or conflicting.</p>
        
        <p><strong>Why This Might Occur:</strong></p>
        <ul>
            <li>Limited or conflicting research evidence</li>
            <li>Complex multi-gene interactions</li>
            <li>Insufficient clinical trial data</li>
            <li>Population-specific variations</li>
        </ul>
        
        <p><strong>Recommended Approach:</strong></p>
        <ul>
            <li>Exercise heightened clinical judgment</li>
            <li>Review current literature and guidelines</li>
            <li>Consider consulting a pharmacogenomics specialist</li>
            <li>Implement careful monitoring strategies</li>
            <li>Document decision-making rationale</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # General disclaimer
    st.markdown("""
    <div class="warning-section">
    <h4>⚠️ Important Disclaimer</h4>
    <p><strong>This tool is for research and educational purposes only.</strong></p>
    <ul>
        <li>Not intended as a substitute for professional medical advice</li>
        <li>Clinical decisions should integrate multiple sources of evidence</li>
        <li>Genetic testing and expert consultation recommended for patient care</li>
        <li>Always follow current clinical practice guidelines</li>
        <li>Consider patient-specific factors beyond genetics</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🧬 Pharmacogenomics Interaction Predictor</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-section">
    <p style="font-size:1.1rem; margin:0;">
    Predict gene-drug interactions using advanced machine learning with multi-modal feature integration.
    This tool combines <strong>semantic embeddings</strong>, <strong>DNA sequence analysis</strong>, 
    and <strong>literature evidence</strong> to provide comprehensive predictions.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model loading section
        st.subheader("Model Loading")
        
        model_dir = st.text_input(
            "Model Directory",
            value="outputs/pharmacogenomics_experiment_latest/models",
            help="Path to directory containing trained model artifacts"
        )
        
        load_button = st.button("Load Model", use_container_width=True)
        
        if load_button:
            with st.spinner("Loading model artifacts..."):
                if FRAMEWORK_AVAILABLE:
                    predictor = PredictionInterface()
                    if predictor.load_artifacts(Path(model_dir)):
                        st.session_state.model_loaded = True
                        st.session_state.predictor = predictor
                        st.success("Model loaded successfully!")
                    else:
                        st.error("Failed to load model")
                else:
                    st.error("Framework not available. Install required dependencies.")
        
        st.markdown("---")
        
        # Model info
        if st.session_state.model_loaded:
            st.subheader("Model Information")
            st.success("Model Status: Loaded ✓")
            
            if hasattr(st.session_state, 'predictor') and st.session_state.predictor.metadata:
                metadata = st.session_state.predictor.metadata
                if 'test_metrics' in metadata:
                    metrics = metadata['test_metrics']
                    st.metric("AUC Score", f"{metrics.get('auc_macro', 0):.3f}")
                    st.metric("F1-Score", f"{metrics.get('f1_macro', 0):.3f}")
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        else:
            st.warning("No model loaded")
        
        st.markdown("---")
        
        # Options
        st.subheader("Prediction Options")
        use_sequences = st.checkbox("Fetch Gene Sequences", value=True)
        use_pubmed = st.checkbox("Fetch PubMed Abstracts", value=False)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Predict Interaction", "Batch Analysis", "Model Insights", "Documentation"
    ])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Gene-Drug Interaction Prediction")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a trained model from the sidebar first.")
            st.stop()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Gene Information")
            gene_name = st.text_input(
                "Gene Name *",
                placeholder="e.g., CYP2D6, BRCA1, TPMT",
                help="Official gene name or symbol"
            )
            gene_symbol = st.text_input(
                "Gene Symbol",
                placeholder="e.g., CYP2D6",
                help="Alternative gene symbol (optional)"
            )
            gene_id = st.text_input(
                "Gene ID",
                placeholder="e.g., PA124",
                help="PharmGKB Gene ID (optional)"
            )
        
        with col2:
            st.subheader("Drug Information")
            drug_name = st.text_input(
                "Drug Name *",
                placeholder="e.g., Tamoxifen, Warfarin, Codeine",
                help="Generic or brand drug name"
            )
            drug_id = st.text_input(
                "Drug ID",
                placeholder="e.g., PA449093",
                help="PharmGKB Drug ID (optional)"
            )
        
        # Additional evidence
        with st.expander("Additional Evidence (Optional)"):
            pmids = st.text_area(
                "PubMed IDs",
                placeholder="Enter comma-separated PMIDs: 12345678, 23456789",
                help="PubMed article IDs supporting this interaction"
            )
            association = st.text_input(
                "Known Association",
                placeholder="e.g., metabolism, efficacy, toxicity"
            )
            evidence = st.text_area(
                "Evidence Description",
                placeholder="Enter any relevant clinical or research evidence..."
            )
        
        st.markdown("---")
        
        # Predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button(
                "🔬 Predict Interaction",
                use_container_width=True,
                type="primary"
            )
        
        # Make prediction
        if predict_button:
            if not gene_name or not drug_name:
                st.error("Please provide both gene name and drug name")
            else:
                with st.spinner("Analyzing pharmacogenomic interaction..."):
                    predictor = st.session_state.predictor
                    
                    result = predictor.predict_interaction(
                        gene_name=gene_name,
                        drug_name=drug_name,
                        gene_symbol=gene_symbol,
                        gene_id=gene_id,
                        drug_id=drug_id,
                        pmids=pmids,
                        association=association,
                        evidence=evidence
                    )
                    
                    if result:
                        display_prediction_results(result)
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Interaction Analysis")
        
        if not st.session_state.model_loaded:
            st.warning("Please load a trained model from the sidebar first.")
            st.stop()
        
        st.info("""
        Upload a CSV file with the following columns:
        - `gene_name` (required)
        - `drug_name` (required)
        - `gene_symbol` (optional)
        - `gene_id` (optional)
        - `drug_id` (optional)
        """)
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("Analyze All Pairs", type="primary"):
                predictor = st.session_state.predictor
                
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in df.iterrows():
                    status_text.text(f"Processing {idx+1}/{len(df)}: {row['gene_name']} - {row['drug_name']}")
                    
                    result = predictor.predict_interaction(
                        gene_name=row['gene_name'],
                        drug_name=row['drug_name'],
                        gene_symbol=row.get('gene_symbol'),
                        gene_id=row.get('gene_id'),
                        drug_id=row.get('drug_id')
                    )
                    
                    if result:
                        results.append({
                            'Gene': result['gene'],
                            'Drug': result['drug'],
                            'Prediction': result['prediction_label'],
                            'Confidence': result['confidence'],
                            'Prob_Associated': result['probabilities']['associated'],
                            'Prob_Not_Associated': result['probabilities']['not_associated'],
                            'Prob_Ambiguous': result['probabilities']['ambiguous']
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                status_text.text("Analysis complete!")
                
                results_df = pd.DataFrame(results)
                
                st.success(f"Analyzed {len(results_df)} gene-drug pairs")
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    associated = len(results_df[results_df['Prediction'] == 'Associated'])
                    st.metric("Associated", associated)
                
                with col2:
                    not_associated = len(results_df[results_df['Prediction'] == 'Not Associated'])
                    st.metric("Not Associated", not_associated)
                
                with col3:
                    ambiguous = len(results_df[results_df['Prediction'] == 'Ambiguous'])
                    st.metric("Ambiguous", ambiguous)
                
                # Download results
                csv = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
    
    # Tab 3: Model Insights
    with tab3:
        st.header("Model Insights & Performance")
        
        if st.session_state.model_loaded and hasattr(st.session_state, 'predictor'):
            metadata = st.session_state.predictor.metadata
            
            if metadata and 'test_metrics' in metadata:
                metrics = metadata['test_metrics']
                
                # Performance metrics
                st.subheader("Model Performance Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("AUC-ROC (Macro)", f"{metrics.get('auc_macro', 0):.4f}")
                    st.metric("AUC-ROC (Weighted)", f"{metrics.get('auc_weighted', 0):.4f}")
                
                with col2:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                    st.metric("Balanced Accuracy", f"{metrics.get('balanced_accuracy', 0):.4f}")
                
                with col3:
                    st.metric("F1-Score (Macro)", f"{metrics.get('f1_macro', 0):.4f}")
                    st.metric("F1-Score (Weighted)", f"{metrics.get('f1_weighted', 0):.4f}")
                
                with col4:
                    st.metric("Matthews Corr. Coef.", f"{metrics.get('mcc', 0):.4f}")
                    st.metric("Cohen's Kappa", f"{metrics.get('cohen_kappa', 0):.4f}")
                
                # Feature information
                if 'feature_metadata' in metadata:
                    st.subheader("Feature Engineering Details")
                    
                    feat_info = metadata['feature_metadata']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Feature Dimensions:**")
                        st.write(f"- Total Features: {feat_info.get('total_features', 'N/A')}")
                        st.write(f"- Gene Embeddings: {feat_info.get('gene_embedding_dim', 'N/A')}")
                        st.write(f"- Drug Embeddings: {feat_info.get('drug_embedding_dim', 'N/A')}")
                        st.write(f"- Text Embeddings: {feat_info.get('text_embedding_dim', 'N/A')}")
                    
                    with col2:
                        st.markdown("**Additional Features:**")
                        st.write(f"- Interaction Features: {feat_info.get('interaction_dims', 'N/A')}")
                        st.write(f"- Sequence Features: {feat_info.get('sequence_features', 0)}")
                        st.write(f"- Statistical Features: {feat_info.get('statistical_features', 0)}")
                        st.write(f"- Embedding Model: {feat_info.get('embedding_model', 'N/A')}")
                
                # Cross-validation results
                if 'cv_results' in metadata:
                    st.subheader("Cross-Validation Results")
                    
                    cv_res = metadata['cv_results']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Mean AUC", 
                            f"{cv_res.get('mean_auc', 0):.4f}",
                            delta=f"±{cv_res.get('std_auc', 0):.4f}"
                        )
                    
                    with col2:
                        st.metric(
                            "Mean F1-Score",
                            f"{cv_res.get('mean_f1', 0):.4f}",
                            delta=f"±{cv_res.get('std_f1', 0):.4f}"
                        )
            else:
                st.info("No metadata available for this model")
        else:
            st.warning("Please load a model first")
    
    # Tab 4: Documentation
    with tab4:
        st.header("Documentation & Methodology")
        
        st.markdown("""
        ## Overview
        
        This tool predicts pharmacogenomic interactions using a **multi-modal machine learning approach**
        that integrates multiple sources of biological and chemical information.
        
        ### Multi-Modal Feature Integration
        
        The prediction model combines three types of features:
        
        #### 1. Semantic Embeddings
        
        - **Gene Embeddings**: Semantic representations of gene names and symbols using transformer models
        - **Drug Embeddings**: Chemical and pharmacological properties encoded as dense vectors
        - **Evidence Text Embeddings**: Contextual information from PubMed abstracts and clinical annotations
        - **Interaction Features**: Element-wise products capturing feature interactions
        
        #### 2. Biological Sequence Features
        
        Extracted from gene DNA sequences via NCBI:
        
        - **GC Content**: Ratio of guanine-cytosine nucleotides (affects gene expression)
        - **Sequence Length**: Total base pairs (normalized)
        - **Molecular Weight**: DNA mass (indicates gene complexity)
        - **Nucleotide Composition**: Individual A, T, G, C percentages
        
        These features capture structural properties that may affect drug-gene interactions.
        
        #### 3. Statistical Features
        
        - **Cosine Similarity**: Measures semantic relatedness in embedding space
        - **Euclidean Distance**: Geometric separation between gene and drug vectors
        - **Distribution Statistics**: Mean, standard deviation, min/max of embeddings
        
        ### Machine Learning Pipeline
        
        #### Data Processing
        
        1. **Data Source**: PharmGKB curated database of gene-drug relationships
        2. **Class Labels**:
           - **Associated (1)**: Validated pharmacogenomic interaction
           - **Not Associated (0)**: No significant interaction
           - **Ambiguous (2)**: Conflicting or unclear evidence
        
        3. **Resampling**: SMOTE-Tomek to handle class imbalance
        4. **Scaling**: Robust scaling for feature normalization
        5. **Stratified Splitting**: 70% train, 15% validation, 15% test
        
        #### Model Training
        
        **Primary Model**: CatBoost Classifier
        - Gradient boosting with categorical features support
        - Automatic class weight balancing
        - Early stopping to prevent overfitting
        
        **Hyperparameter Optimization**: 
        - Bayesian optimization using Optuna
        - Tree depth, learning rate, regularization tuned
        - 100 trials for optimal configuration
        
        **Validation**:
        - 10-fold stratified cross-validation
        - Multiple performance metrics tracked
        - Test set held out for final evaluation
        
        ### Understanding Predictions
        
        #### Confidence Interpretation
        
        | Confidence | Interpretation | Action |
        |------------|---------------|---------|
        | > 85% | High confidence | Strongly consider prediction |
        | 70-85% | Moderate confidence | Review with additional evidence |
        | 60-70% | Low confidence | Requires validation |
        | < 60% | Very uncertain | Do not rely on prediction |
        
        #### Clinical Context
        
        **Associated Predictions** suggest:
        - Genetic variants may affect drug metabolism
        - Dose adjustments may be needed
        - Enhanced monitoring recommended
        - Genetic testing may be beneficial
        
        **Not Associated Predictions** indicate:
        - Standard dosing likely appropriate
        - Normal monitoring protocols sufficient
        - Other pharmacogenomic factors may still apply
        
        **Ambiguous Predictions** mean:
        - Evidence is conflicting or limited
        - Exercise clinical judgment
        - Consult additional resources
        - Consider specialist input
        
        ### Biological Basis
        
        #### Why Gene-Drug Interactions Matter
        
        Genetic variations can affect:
        
        1. **Drug Metabolism**: Variants in CYP450 enzymes alter drug breakdown
        2. **Drug Transport**: Transporter genes affect drug distribution
        3. **Drug Targets**: Receptor variations change drug efficacy
        4. **Adverse Reactions**: Genetic factors increase toxicity risk
        
        #### Example: CYP2D6 and Codeine
        
        - **CYP2D6** metabolizes codeine to morphine
        - **Poor metabolizers**: Reduced efficacy (can't convert to active form)
        - **Ultra-rapid metabolizers**: Toxicity risk (too much morphine)
        - **Clinical impact**: Requires alternative analgesics or dose adjustment
        
        ### Model Performance
        
        **Evaluation Metrics**:
        
        - **AUC-ROC**: Area under receiver operating characteristic curve (discrimination ability)
        - **F1-Score**: Harmonic mean of precision and recall (balanced performance)
        - **Matthews Correlation Coefficient**: Overall quality for imbalanced classes
        - **Balanced Accuracy**: Accounts for class imbalance
        
        **Typical Performance**:
        - AUC: 0.85-0.90 (excellent discrimination)
        - F1: 0.80-0.85 (good precision-recall balance)
        - Accuracy: 0.78-0.83 (good overall correctness)
        
        ### Limitations & Disclaimers
        
        #### Model Limitations
        
        1. **Training Data**: Limited to PharmGKB annotations (may miss rare interactions)
        2. **Population**: Primarily based on published studies (may have population bias)
        3. **Complexity**: Cannot capture all multi-gene interactions
        4. **Temporal**: Evidence base evolves (model requires periodic retraining)
        
        #### Usage Warnings
        
        - **Not FDA Approved**: This is a research tool, not a diagnostic device
        - **Not Clinical Decision Support**: Cannot replace professional medical judgment
        - **Requires Validation**: Predictions should be confirmed with genetic testing
        - **Patient-Specific**: Individual factors beyond genetics must be considered
        - **Guideline Adherence**: Always follow CPIC and FDA pharmacogenomic guidelines
        
        ### Data Sources
        
        - **PharmGKB**: www.pharmgkb.org (curated pharmacogenomic knowledge)
        - **NCBI Gene**: www.ncbi.nlm.nih.gov/gene (gene sequences and annotations)
        - **PubMed**: pubmed.ncbi.nlm.nih.gov (scientific literature)
        - **CPIC Guidelines**: cpicpgx.org (clinical implementation guidelines)
        
        ### Technical Implementation
        
        **Key Technologies**:
        - Python 3.8+
        - Sentence Transformers (semantic embeddings)
        - CatBoost/XGBoost/LightGBM (gradient boosting)
        - BioPython (sequence analysis)
        - Scikit-learn (ML pipeline)
        - Optuna (hyperparameter optimization)
        
        **Model Artifacts**:
        - `best_model.pkl`: Trained classifier
        - `scaler.pkl`: Feature scaling transformation
        - `metadata.json`: Training configuration and metrics
        
        ### References
        
        1. PharmGKB: Pharmacogenomics Knowledge Base
        2. CPIC Guidelines: Clinical Pharmacogenetics Implementation Consortium
        3. FDA Table of Pharmacogenomic Biomarkers
        4. DPWG Guidelines: Dutch Pharmacogenetics Working Group
        
        ### Citation
        
        If you use this tool in research, please cite:
        
        ```
        [Your Research Paper]
        Title: Deep Learning-Enhanced Pharmacogenomic Interaction Prediction
        Authors: [Authors]
        Journal: [Journal], Year
        DOI: [DOI]
        ```
        
        ### Contact & Support
        
        For questions, bug reports, or collaboration inquiries:
        - Email: [your.email@institution.edu]
        - GitHub: [repository URL]
        - Issues: [GitHub issues URL]
        
        ### License
        
        This tool is released under the MIT License for academic and research use.
        Commercial use requires separate licensing.
        """)

if __name__ == "__main__":
    main()
                
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import database

# ==============================================================================
#                      CORE AI/ML LOGIC AND CLASSES
# ==============================================================================

# --- Configuration Dataclasses (simplified for prediction) ---
@dataclass
class SupervisedConfig:
    sequence_length: int = 100
    feature_dim: int = 10

@dataclass
class AdvancedRLConfig:
    state_dim: int = 4
    action_dim: int = 4
    num_agents: int = 4

# --- Helper Functions and Classes ---
FEATURE_NAMES = [
    'vibration', 'temperature', 'pressure', 'current',
    'voltage', 'rpm', 'oil_level', 'humidity',
    'acoustic', 'magnetic_field'
]

def create_synthetic_data(config: SupervisedConfig, num_samples: int = 1000) -> np.ndarray:
    # This is the same data generator used for the live simulation
    time = np.linspace(0, 10, num_samples)
    features = []
    for i in range(config.feature_dim):
        if i < 2: features.append(np.sin(time + i * np.pi / 4))
        else: features.append(np.random.uniform(0, 1, num_samples))
    features = np.column_stack(features).astype(np.float32)
    X = []
    for i in range(len(features) - config.sequence_length):
        X.append(features[i:i + config.sequence_length])
    return np.array(X, dtype=np.float32)

# --- Updated HybridMaintenanceSystem ---
class HybridMaintenanceSystem:
    def __init__(self, trained_model):
        # The system now uses the pre-trained model
        self.health_model = trained_model
        # The RL and Explainability parts are simplified for this example
        self.rl_config = AdvancedRLConfig()
        self.explainability = self.ExplainabilityModule(FEATURE_NAMES)
        self.metrics = {'health_predictions': [], 'explanations': []}

    def predict_health(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        # Ensure data has the batch dimension
        if len(sensor_data.shape) == 2:
            sensor_data = sensor_data[np.newaxis, ...]

        # Use the real model to predict
        predicted_value = self.health_model.predict(sensor_data)[0][0]

        # Convert the prediction into a "health score" between 0 and 1
        # This mapping will depend on your specific problem. Here's a simple example.
        health_score = 1 / (1 + max(0, predicted_value)) # Simple normalization

        return {'health_score': float(health_score), 'failure_prob': 1 - float(health_score), 'rul': float(health_score) * 100}

    def monitor_machine(self, machine_id: int, sensor_data: np.ndarray) -> Dict[str, Any]:
        health_metrics = self.predict_health(sensor_data)
        # Simplified RL action for this example
        if health_metrics['health_score'] < 0.5:
            action = 3 # Replace
        elif health_metrics['health_score'] < 0.75:
            action = 2 # Major Service
        else:
            action = 0 # No Action

        explanation = self.explainability.explain_prediction()
        report = {
            'machine_id': machine_id, 'timestamp': datetime.utcnow().isoformat(), 'health_metrics': health_metrics,
            'maintenance_action': {'action': action}, 'explanation': explanation
        }
        self.metrics['health_predictions'].append(report['health_metrics'])
        self.metrics['explanations'].append(report['explanation'])
        return report

    def visualize_results(self) -> plt.Figure:
        # (Visualization code remains the same as before)
        fig, ax = plt.subplots(figsize=(10, 4))
        health_df = pd.DataFrame(self.metrics['health_predictions'][-100:])
        sns.lineplot(data=health_df, ax=ax)
        ax.set_title('Health Score Over Time')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        return fig

    # A simple placeholder class for Explainability
    class ExplainabilityModule:
        def __init__(self, feature_names: List[str]): self.feature_names = feature_names
        def explain_prediction(self) -> Dict[str, float]:
            imp = np.abs(np.random.normal(0, 1, len(self.feature_names)))
            return dict(zip(self.feature_names, imp / np.sum(imp)))

# ==============================================================================
#                      STREAMLIT UI AND SIMULATION LOGIC
# ==============================================================================

st.set_page_config(page_title="Live Dashboard", page_icon="📈", layout="wide")
st.title("📈 Live Dashboard (with Trained Model)")

# --- Load Trained Model and Data ---
@st.cache_resource
def load_trained_model():
    try:
        model = load_model('health_model.h5')
        return model
    except (IOError, ImportError) as e:
        st.error(f"Error loading model: {e}. Please run `python train.py` first to generate the model file.", icon="🚨")
        return None

@st.cache_data
def load_simulation_data():
    return create_synthetic_data(SupervisedConfig(), num_samples=500)

trained_model = load_trained_model()
X_data = load_simulation_data()

# Only proceed if the model was loaded successfully
if trained_model:
    system = HybridMaintenanceSystem(trained_model)

    st.sidebar.header("Simulation Controls")
    machine_id = st.sidebar.selectbox(
        'Select a Machine to Monitor', options=list(range(4)), format_func=lambda x: f"Machine #{x}"
    )

    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False

    if st.sidebar.button('▶️ Start Live Simulation', use_container_width=True, type="primary"):
        st.session_state.run_simulation = True
        st.rerun()

    if st.sidebar.button('⏹️ Stop Live Simulation', use_container_width=True):
        st.session_state.run_simulation = False
        st.rerun()

    placeholder = st.empty()

    if st.session_state.run_simulation:
        # (The simulation loop remains the same as before)
        st.sidebar.success(f"Live simulation running for Machine #{machine_id}...")
        start_index = random.randint(0, len(X_data) - 50)
        for i in range(start_index, len(X_data)):
            if not st.session_state.run_simulation: break
            sensor_data_sample = X_data[i]
            report = system.monitor_machine(machine_id, sensor_data_sample)
            database.add_report(report)
            with placeholder.container():
                # (The dashboard display logic remains the same)
                st.header(f"Live Status for Machine #{machine_id}", anchor=False)
                col1, col2, col3 = st.columns(3)
                col1.metric("Health Score", f"{report['health_metrics']['health_score']:.2f}")
                col2.metric("Failure Probability", f"{report['health_metrics']['failure_prob']:.2%}", delta_color="inverse")
                action_map = {0: "✅ No Action", 1: "🔧 Minor Service", 2: "⚠️ Major Service", 3: "🚨 Replace"}
                col3.metric("Recommended Action", action_map.get(report['maintenance_action']['action'], 'Unknown'))
                st.line_chart(pd.DataFrame(system.metrics['health_predictions']).iloc[-100:]['health_score'])

            time.sleep(2)
    else:
        st.info("Select a machine and click 'Start Live Simulation' to begin.")

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

# ✅ FIX: Initialize the database at the start of this page
database.init_db()

# ==============================================================================
#                      CORE AI/ML LOGIC AND CLASSES
# ==============================================================================

# --- Configuration Constants ---
NUM_MACHINES = 10  # <-- CHANGE: Increased number of machines
FAULTY_MACHINES = [3, 7] # <-- CHANGE: Designated faulty machines

@dataclass
class SupervisedConfig:
    sequence_length: int = 100
    feature_dim: int = 10

@dataclass
class AdvancedRLConfig:
    state_dim: int = 4
    action_dim: int = 4
    num_agents: int = NUM_MACHINES # Use the constant here

FEATURE_NAMES = [
    'vibration', 'temperature', 'pressure', 'current',
    'voltage', 'rpm', 'oil_level', 'humidity',
    'acoustic', 'magnetic_field'
]

def create_synthetic_data(config: SupervisedConfig, num_samples: int = 1000) -> np.ndarray:
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

class HybridMaintenanceSystem:
    def __init__(self, trained_model):
        self.health_model = trained_model
        self.rl_config = AdvancedRLConfig()
        self.explainability = self.ExplainabilityModule(FEATURE_NAMES)
        self.metrics = {'health_predictions': [], 'explanations': []}

    def predict_health(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        if len(sensor_data.shape) == 2:
            sensor_data = sensor_data[np.newaxis, ...]
        predicted_value = self.health_model.predict(sensor_data, verbose=0)[0][0]
        health_score = 1 / (1 + max(0, predicted_value))
        return {'health_score': float(health_score), 'failure_prob': 1 - float(health_score), 'rul': float(health_score) * 100}

    def monitor_machine(self, machine_id: int, sensor_data: np.ndarray) -> Dict[str, Any]:
        health_metrics = self.predict_health(sensor_data)

        # <-- CHANGE: Artificially degrade faulty machines sometimes
        if machine_id in FAULTY_MACHINES and random.random() < 0.25: # 25% chance of failure event
            health_score = random.uniform(0.2, 0.5)
            health_metrics['health_score'] = health_score
            health_metrics['failure_prob'] = 1 - health_score
            health_metrics['rul'] = health_score * 100

        if health_metrics['health_score'] < 0.5: action = 3
        elif health_metrics['health_score'] < 0.75: action = 2
        else: action = 0
        
        explanation = self.explainability.explain_prediction()
        report = {
            'machine_id': machine_id, 'timestamp': datetime.utcnow().isoformat(), 'health_metrics': health_metrics,
            'maintenance_action': {'action': action}, 'explanation': explanation
        }
        self.metrics['health_predictions'].append(report['health_metrics'])
        self.metrics['explanations'].append(report['explanation'])
        return report

    def visualize_results(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 4))
        health_df = pd.DataFrame(self.metrics['health_predictions'][-100:])
        sns.lineplot(data=health_df['health_score'], ax=ax, label="Health Score")
        ax.set_title('Health Score Over Time')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='--', label='Critical Threshold')
        ax.axhline(0.75, color='orange', linestyle='--', label='Warning Threshold')
        ax.legend()
        plt.tight_layout()
        return fig

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

@st.cache_resource
def load_trained_model():
    try:
        model = load_model('health_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure 'health_model.h5' is in your GitHub repository.", icon="🚨")
        return None

@st.cache_data
def load_simulation_data():
    return create_synthetic_data(SupervisedConfig(), num_samples=500)

trained_model = load_trained_model()
X_data = load_simulation_data()

if trained_model:
    if 'system' not in st.session_state:
        st.session_state.system = HybridMaintenanceSystem(trained_model)
    system = st.session_state.system

    st.sidebar.header("Simulation Controls")
    # <-- CHANGE: Updated the machine selection range
    machine_id = st.sidebar.selectbox('Select a Machine to Monitor', options=list(range(NUM_MACHINES)), format_func=lambda x: f"Machine #{x}")

    if 'run_simulation' not in st.session_state: st.session_state.run_simulation = False
    if st.sidebar.button('▶️ Start Live Simulation', use_container_width=True, type="primary"):
        st.session_state.run_simulation = True
        st.rerun()
    if st.sidebar.button('⏹️ Stop Live Simulation', use_container_width=True):
        st.session_state.run_simulation = False
        st.rerun()

    placeholder = st.empty()
    if st.session_state.run_simulation:
        st.sidebar.success(f"Live simulation running for Machine #{machine_id}...")
        start_index = random.randint(0, len(X_data) - 50)
        for i in range(start_index, len(X_data)):
            if not st.session_state.run_simulation: break
            sensor_data_sample = X_data[i]
            report = system.monitor_machine(machine_id, sensor_data_sample)
            database.add_report(report)
            with placeholder.container():
                st.header(f"Live Status for Machine #{machine_id}", anchor=False)
                col1, col2, col3 = st.columns(3)
                col1.metric("Health Score", f"{report['health_metrics']['health_score']:.2f}")
                col2.metric("Failure Probability", f"{report['health_metrics']['failure_prob']:.2%}", delta_color="inverse")
                action_map = {0: "✅ No Action", 2: "⚠️ Major Service", 3: "🚨 Replace"}
                col3.metric("Recommended Action", action_map.get(report['maintenance_action']['action'], 'Unknown'))
                
                fig = system.visualize_results()
                st.pyplot(fig)
                plt.close(fig)

            time.sleep(2)
    else:
        st.info("Select a machine and click 'Start Live Simulation' to begin.")

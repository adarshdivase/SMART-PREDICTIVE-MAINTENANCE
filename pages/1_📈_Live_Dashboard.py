import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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

@dataclass
class SupervisedConfig:
    sequence_length: int = 100
    feature_dim: int = 10

@dataclass
class AdvancedRLConfig:
    state_dim: int = 4
    action_dim: int = 4
    hidden_units: int = 256
    learning_rate: float = 0.0003
    num_agents: int = 4

@dataclass
class HybridSystemConfig:
    model_checkpoint_dir: str = "hybrid_checkpoints"

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
        elif i < 4: features.append(np.sin((i + 1) * time))
        else: features.append(np.random.uniform(0, 1, num_samples))
    features = np.column_stack(features).astype(np.float32)
    X = []
    for i in range(len(features) - config.sequence_length):
        X.append(features[i:i + config.sequence_length])
    return np.array(X, dtype=np.float32)

class PPOActorCritic(Model):
    def __init__(self, config: AdvancedRLConfig):
        super().__init__()
        self.shared_layers = [Dense(config.hidden_units, activation='relu'), Dense(config.hidden_units, activation='relu')]
        self.policy_layers = [Dense(config.action_dim, activation='softmax')]
        self.value_layers = [Dense(1)]
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = inputs
        for layer in self.shared_layers: x = layer(x)
        policy = x
        for layer in self.policy_layers: policy = layer(policy)
        value = x
        for layer in self.value_layers: value = layer(value)
        return policy, value

class MultiAgentPPO:
    def __init__(self, config: AdvancedRLConfig):
        self.config = config
        self.agents = [PPOActorCritic(config) for _ in range(config.num_agents)]
    def get_action(self, agent_id: int, state: np.ndarray) -> Tuple[int, float, np.ndarray]:
        state = tf.cast(state[np.newaxis, :], tf.float32)
        policy, value = self.agents[agent_id](state)
        action = tf.random.categorical(tf.math.log(policy), 1)[0, 0]
        return int(action), float(value[0, 0]), policy[0].numpy()

class ExplainabilityModule:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
    def explain_prediction(self) -> Dict[str, float]:
        importance_scores = np.abs(np.random.normal(0, 1, len(self.feature_names)))
        return dict(zip(self.feature_names, importance_scores / np.sum(importance_scores)))

class HybridMaintenanceSystem:
    def __init__(self, supervised_config: SupervisedConfig, rl_config: AdvancedRLConfig, hybrid_config: HybridSystemConfig):
        self.supervised_config = supervised_config
        self.rl_config = rl_config
        self.multi_agent_system = MultiAgentPPO(rl_config)
        self.explainability = ExplainabilityModule(FEATURE_NAMES)
        self.metrics = {'health_predictions': [], 'maintenance_decisions': [], 'explanations': []}

    def predict_health(self, sensor_data: np.ndarray) -> Dict[str, Any]:
        health_score = tf.sigmoid(tf.reduce_mean(sensor_data[:, :, 0]))
        return {'health_score': float(health_score), 'failure_prob': 1 - float(health_score), 'rul': float(health_score) * 100}

    def monitor_machine(self, machine_id: int, sensor_data: np.ndarray) -> Dict[str, Any]:
        # ✅ THE FIX IS HERE: Add a batch dimension before predicting
        if len(sensor_data.shape) == 2:
            sensor_data_batch = sensor_data[np.newaxis, ...] # Converts shape (100, 10) to (1, 100, 10)
        else:
            sensor_data_batch = sensor_data

        # Use the new 3D variable for prediction
        health_metrics = self.predict_health(sensor_data_batch)

        # The rest of the function remains the same
        state = np.array(list(health_metrics.values()), dtype=np.float32)
        action, value, policy = self.multi_agent_system.get_action(machine_id % self.rl_config.num_agents, state)
        explanation = self.explainability.explain_prediction()
        report = {
            'machine_id': machine_id, 'timestamp': datetime.utcnow().isoformat(), 'health_metrics': health_metrics,
            'maintenance_action': {'action': action, 'value': value, 'confidence': float(np.max(policy))}, 'explanation': explanation
        }
        self.metrics['health_predictions'].append(report['health_metrics'])
        self.metrics['maintenance_decisions'].append(report['maintenance_action'])
        self.metrics['explanations'].append(report['explanation'])
        return report

    def visualize_results(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        health_df = pd.DataFrame(self.metrics['health_predictions'][-100:])
        sns.lineplot(data=health_df, ax=axes[0])
        axes[0].set_title('Health Metrics Over Time')
        latest_explanation = self.metrics['explanations'][-1] if self.metrics['explanations'] else {}
        if latest_explanation:
            sns.barplot(x=list(latest_explanation.values()), y=list(latest_explanation.keys()), ax=axes[1], orient='h')
        axes[1].set_title('Latest Feature Importance')
        plt.tight_layout()
        return fig

# ==============================================================================
#                      STREAMLIT UI AND SIMULATION LOGIC
# ==============================================================================

st.set_page_config(page_title="Live Dashboard", page_icon="📈", layout="wide")
st.title("📈 Live Monitoring Dashboard")
st.markdown("Start the simulation to see real-time predictions and save results to the database.")

@st.cache_data
def load_data():
    return create_synthetic_data(SupervisedConfig(), num_samples=500)

@st.cache_resource
def load_system():
    return HybridMaintenanceSystem(SupervisedConfig(), AdvancedRLConfig(), HybridSystemConfig())

X_data = load_data()
system = load_system()

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
    st.sidebar.success(f"Live simulation running for Machine #{machine_id}...")
    start_index = random.randint(0, len(X_data) - 50)
    for i in range(start_index, len(X_data)):
        if not st.session_state.run_simulation:
            st.sidebar.warning("Simulation stopped.")
            break

        sensor_data_sample = X_data[i]
        report = system.monitor_machine(machine_id, sensor_data_sample)
        database.add_report(report)

        with placeholder.container():
            st.header(f"Live Status for Machine #{machine_id}", anchor=False)
            st.write(f"Last update: {datetime.now().strftime('%I:%M:%S %p')}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Health Score", f"{report['health_metrics']['health_score']:.2f}")
            col2.metric("Failure Probability", f"{report['health_metrics']['failure_prob']:.2%}", delta_color="inverse")
            col3.metric("RUL (hours)", f"{report['health_metrics']['rul']:.1f}")
            st.divider()

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Recommended Action", anchor=False)
                action_map = {0: "✅ No Action", 1: "🔧 Minor Service", 2: "⚠️ Major Service", 3: "🚨 Replace"}
                st.info(f"**Action:** {action_map.get(report['maintenance_action']['action'], 'Unknown')}")

            with col2:
                st.subheader("Top Contributing Factors", anchor=False)
                st.bar_chart(report['explanation'], height=250)

            st.subheader("Live Trend", anchor=False)
            fig = system.visualize_results()
            st.pyplot(fig)
            plt.close(fig)

        time.sleep(3)
else:
    st.info("Select a machine and click 'Start Live Simulation' to begin.")

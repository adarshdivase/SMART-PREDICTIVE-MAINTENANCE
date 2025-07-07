# ⚙️ Hybrid Predictive Maintenance System

This project is a multi-page Streamlit web application that demonstrates a hybrid predictive maintenance system. It uses a combination of supervised learning and multi-agent reinforcement learning to predict machine health and recommend optimal maintenance actions.

## Features

- **Multi-Page Dashboard:** A clean interface separating the live simulation from historical analysis.
- **Live Monitoring:** Run a real-time simulation to monitor machine health metrics.
- **AI-Powered Recommendations:** Get maintenance action recommendations from a reinforcement learning agent.
- **Persistent History:** All simulation results are saved to a local SQLite database.
- **Historical Explorer:** View and analyze past performance, predictions, and maintenance records for any machine.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

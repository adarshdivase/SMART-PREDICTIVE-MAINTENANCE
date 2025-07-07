import streamlit as st
import database

# Set the page configuration for the entire app
st.set_page_config(
    page_title="Hybrid Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# Initialize the database which creates the .db file on first run
database.init_db()

# --- Page Content ---
st.title("⚙️ Welcome to the Hybrid Predictive Maintenance System")
st.markdown("""
This application demonstrates a system for predicting machine health and recommending optimal maintenance actions using a hybrid AI model.

**Navigate to the pages in the sidebar to get started:**

- **`Live Dashboard`:** Run a real-time simulation of machine monitoring and see AI-driven recommendations as they happen.
- **`Historical Explorer`:** View and analyze past performance, predictions, and maintenance records for any machine.

This system saves all simulation results to a local database, allowing you to track performance over time.
""")

st.info("This system uses a combination of supervised learning for health prediction and reinforcement learning for decision-making.", icon="🤖")

st.success("To begin, select a page from the sidebar on the left.")

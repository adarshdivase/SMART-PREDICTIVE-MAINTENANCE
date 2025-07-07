import streamlit as st

st.set_page_config(
    page_title="Hybrid Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

# The database.init_db() call has been removed from this file.

st.title("⚙️ Welcome to the Hybrid Predictive Maintenance System")
st.markdown("""
This application demonstrates a system for predicting machine health
and recommending optimal maintenance actions using a hybrid AI model.

**Navigate to the pages in the sidebar to get started:**

- **`Live Dashboard`:** Run a real-time simulation of machine monitoring.
- **`Historical Explorer`:** View and analyze past performance and records.
""")

st.info("This system uses a combination of supervised learning for health prediction and reinforcement learning for decision-making.", icon="🤖")

st.success("To begin, select a page from the sidebar on the left.")

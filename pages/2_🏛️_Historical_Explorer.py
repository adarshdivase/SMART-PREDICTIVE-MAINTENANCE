import streamlit as st
import pandas as pd
import database

# ✅ FIX: Initialize the database at the start of this page
database.init_db()

st.set_page_config(page_title="Historical Explorer", page_icon="🏛️", layout="wide")
st.title("🏛️ Historical Data Explorer")
st.markdown("Analyze past performance and maintenance records for any machine in the system.")

st.sidebar.header("Analysis Options")
machine_id = st.sidebar.selectbox(
    'Select a Machine to Analyze',
    options=list(range(4)),
    format_func=lambda x: f"Machine #{x}"
)

st.divider()

history_df = database.get_reports_by_machine(machine_id)

if history_df.empty:
    st.warning(f"No historical data found for Machine #{machine_id}. Please run the live simulation on the 'Live Dashboard' page first.")
else:
    st.header(f"Performance History for Machine #{machine_id}", anchor=False)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

    col1, col2, col3 = st.columns(3)
    avg_health = history_df['health_score'].mean()
    total_reports = len(history_df)
    last_action_code = history_df['action'].iloc[-1] if not history_df.empty else "N/A"
    col1.metric("Total Records Logged", total_reports)
    col2.metric("Average Health Score", f"{avg_health:.2f}")
    col3.metric("Last Action Code", f"{last_action_code}")
    st.divider()

    st.subheader("Health Metrics Over Time")
    st.line_chart(history_df, x='timestamp', y=['health_score', 'failure_prob'], color=["#00FF00", "#FF0000"])

    st.subheader("History of Recommended Actions")
    action_map = {0: "No Action", 1: "Minor Service", 2: "Major Service", 3: "Replace"}
    action_counts = history_df['action'].value_counts().rename(index=action_map)
    st.bar_chart(action_counts)
    st.divider()

    st.subheader("Raw Data Log")
    st.dataframe(
        history_df,
        column_config={
            "timestamp": st.column_config.DatetimeColumn("Timestamp", format="YYYY-MM-DD HH:mm:ss"),
            "health_score": st.column_config.ProgressColumn("Health Score", min_value=0, max_value=1),
            "failure_prob": st.column_config.ProgressColumn("Failure Probability", min_value=0, max_value=1),
        },
        use_container_width=True
    )

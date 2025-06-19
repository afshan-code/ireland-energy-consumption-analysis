import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ---------- Load Saved Results ---------- #
loaded_data = joblib.load("ml_results.pkl")
results = loaded_data["results"]
predictions = loaded_data["predictions"]
y_test_original = loaded_data["y_test_original"]
cv_results = loaded_data["cv_results"]

# ---------- Streamlit App Setup ---------- #
st.set_page_config(page_title="Energy ML Dashboard", layout="wide")
st.title("Energy ML Model Dashboard")

st.markdown("""
This dashboard presents the performance of supervised ML models — **Decision Tree** and **Random Forest** — used in energy prediction.  
It follows Edward Tufte's principles: **data density**, **minimal chartjunk**, and **clear comparisons**.
""")

# ---------- Section 1: Model Evaluation Metrics ---------- #
st.header("1.Model Evaluation Metrics")

metrics_df = results.copy()
tab1, tab2 = st.tabs(["Metrics Table", "Metric Comparison"])
numeric_columns = metrics_df.select_dtypes(include=np.number).columns.tolist()



with tab1:
    model_filter = st.radio(
    "Select model(s):", 
    ["Decision Tree", "Random Forest", "Both"], 
    horizontal=True,
    key="metrics_model_filter")


    if model_filter != "Both":
        df_filtered = metrics_df[metrics_df["Model"] == model_filter]
        numeric_df = df_filtered.select_dtypes(include='number')  # select numeric columns only
        st.dataframe(numeric_df.style.highlight_max(axis=1), use_container_width=True)
    else:
        # subset numeric metric columns explicitly for highlight_max
        st.dataframe(
            metrics_df.style.highlight_max(
                axis=0, subset=["R² Score (Test)", "R² Score (Train)", "MAE", "RMSE"]
            ),
            use_container_width=True,
        )

with tab2:
    #selected_metric = st.selectbox("Choose a metric to compare:", metrics_df.columns)
    #plot_df = metrics_df[["Model", selected_metric]].reset_index(drop=True)

    selected_metric = st.selectbox("Choose a metric to compare:", numeric_columns)
    plot_df = metrics_df[["Model", selected_metric]].reset_index(drop=True)

    fig = px.bar(
         plot_df,
    x="Model",
    y=selected_metric,
    color="Model",
    color_discrete_map={"Decision Tree": "#8ecae6", "Random Forest": "#ffb703"},
    text=selected_metric,
    title=f"{selected_metric} Comparison Between Models"
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_title=selected_metric,
        xaxis_title="Model",
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="black"),
        title_font=dict(size=18),
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------- Section 2: Actual vs Predicted (Log-Log Scale) ---------- #
st.header("2.Actual vs Predicted (Log-Log Scale)")

highlight_model = st.radio(
    "Highlight model:", 
    ["Decision Tree", "Random Forest", "Both"], 
    horizontal=True,
    key="highlight_model_radio"
)


def get_log_ticks(min_val, max_val):
    exp_range = range(int(np.floor(np.log10(min_val))), int(np.ceil(np.log10(max_val))) + 1)
    return [10**i for i in exp_range], [f"10^{i}" for i in exp_range]

def create_loglog_plot(actual, predicted, model, opacity=1.0, disable_hover=False):
    actual = np.array(actual)
    predicted = np.array(predicted)
    valid = (actual > 0) & (predicted > 0)
    actual, predicted = actual[valid], predicted[valid]

    min_val, max_val = max(min(min(actual), min(predicted)), 1e-1), max(max(actual), max(predicted))
    ticks, labels = get_log_ticks(min_val, max_val)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(color='blue', size=6, opacity=opacity),
        name=model, showlegend=False,
        hoverinfo='skip' if disable_hover else 'text',
        text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}" for a, p in zip(actual, predicted)] if not disable_hover else None
    ))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line=dict(color='red', dash='dash'), showlegend=False
    ))
    fig.update_layout(
        title=f"{model}: Actual vs Predicted (Log-Log Scale)",
        xaxis=dict(title="Actual VALUE (log scale)", type="log", tickvals=ticks, ticktext=labels),
        yaxis=dict(title="Predicted VALUE (log scale)", type="log", tickvals=ticks, ticktext=labels, scaleanchor="x", scaleratio=1),
        plot_bgcolor='white', paper_bgcolor='white', height=500
    )
    return fig

col1, col2 = st.columns(2)

if highlight_model == "Both":
    with col1:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Decision Tree"], "Decision Tree"), use_container_width=True)
    with col2:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Random Forest"], "Random Forest"), use_container_width=True)
elif highlight_model == "Decision Tree":
    with col1:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Decision Tree"], "Decision Tree"), use_container_width=True)
    with col2:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Random Forest"], "Random Forest", opacity=0.15, disable_hover=True), use_container_width=True)
else:
    with col1:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Decision Tree"], "Decision Tree", opacity=0.15, disable_hover=True), use_container_width=True)
    with col2:
        st.plotly_chart(create_loglog_plot(y_test_original, predictions["Random Forest"], "Random Forest"), use_container_width=True)

# ---------- Section 3: Cross-Validation Summary ---------- #

st.header("3.Cross-Validation Results")

cv_df = pd.DataFrame(cv_results)

cv_choice = st.radio(
    "Select model(s):", 
    ["Decision Tree", "Random Forest"], 
    horizontal=True,
    key="cv_model_choice"
)

# Filter based on user choice while keeping the 'Model' column visible
if cv_choice != "Both":
    cv_filtered = cv_df[cv_df["Model"] == cv_choice]
else:
    cv_filtered = cv_df

# Display the filtered table with model column included
st.dataframe(cv_filtered, use_container_width=True)



# ---------- Section 4: Download Center ---------- #
st.header("4.Download Results")

# Download Metrics
st.download_button(
    "Download Metrics CSV",
    data=metrics_df.to_csv(index=False).encode("utf-8"),
    file_name="model_metrics.csv",
    mime="text/csv"
)

# Download Predictions
for model_name, y_pred in predictions.items():
    df_pred = pd.DataFrame({"Actual": y_test_original, "Predicted": y_pred})
    st.download_button(
        f"Download Predictions - {model_name}",
        data=df_pred.to_csv(index=False).encode("utf-8"),
        file_name=f"{model_name}_predictions.csv",
        mime="text/csv"
    )

# Download Cross-Validation
st.download_button(
    "Download CV Results CSV",
    data=cv_df.to_csv(index=False).encode("utf-8"),
    file_name="cv_summary.csv",
    mime="text/csv"
)

# ---------- Footer ---------- #
st.markdown("---")
st.caption("Developed for Energy Analytics — Using Edward Tufte's Visualization Principles: Data-rich, Minimalist, Comparative.")

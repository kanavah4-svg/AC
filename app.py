import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px

# Custom colour palette for charts
PALETTE = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]

@st.cache_data
def load_data():
    df = pd.read_csv("atelier8_survey_data.csv")
    return df

def main():
    st.set_page_config(
        page_title="ATELIER 8 – Circular Luxury Intelligence Dashboard",
        page_icon="👜",
        layout="wide"
    )

    # ---------- HEADER ----------
    st.markdown("<h1 style='margin-bottom:0px;'>ATELIER 8 – Circular Luxury Intelligence Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#555; font-size:15px; margin-top:4px;'>"
        "Data-driven insights for restoration, authentication, and circular luxury in the UAE."
        "</p>",
        unsafe_allow_html=True
    )

    with st.expander("📌 What is this dashboard showing?", expanded=True):
        st.write(
            "- **Business idea**: ATELIER 8 is a circular luxury restoration & authentication studio for handbags and sneakers.\n"
            "- **Dataset**: Synthetic survey of 400 potential UAE luxury consumers (age, income, ownership, sustainability, willingness-to-pay).\n"
            "- **Goal**: Help a non-technical viewer understand

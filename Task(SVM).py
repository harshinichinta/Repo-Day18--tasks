import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Breast Cancer Data Analysis App")

# File uploader (instead of hardcoded path)
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Show dataset
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.subheader("Dataset Info")
    st.write(df.describe())

    # Column selection
    column = st.selectbox("Select column for analysis", df.columns)

    # Histogram
    st.subheader("Histogram")
    fig, ax = plt.subplots()
    ax.hist(df[column])
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots()
    sns.heatmap(df.corr(), annot=True, ax=ax2)
    st.pyplot(fig2)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit app title
st.title("Housing Market Data Visualizer")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with housing market data", type=["csv"])

if uploaded_file is not None:
    # Read CSV file
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data:")
    st.dataframe(df.head())

    # Select numeric columns for visualization
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found in the dataset!")
    else:
        # Select columns for visualization
        x_axis = st.selectbox("Select X-axis variable", numeric_cols)
        y_axis = st.selectbox("Select Y-axis variable", numeric_cols)

        # Filter data based on user input
        st.write("### Data Filtering")
        filter_col = st.selectbox("Select a column to filter (optional)", [None] + df.columns.tolist())

        if filter_col:
            unique_values = df[filter_col].dropna().unique()
            selected_values = st.multiselect("Select values to include", unique_values, default=unique_values[:5])
            df = df[df[filter_col].isin(selected_values)]

        # Create scatter plot
        st.write("### Housing Market Trends")
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.5)
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f"{y_axis} vs {x_axis}")
        st.pyplot(fig)

        # Show summary statistics
        st.write("### Summary Statistics")
        st.write(df.describe())


import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import (
    load_data, basic_cleaning, impute_missing,
    engineer_features, create_investment_label,
    predict_future_price
)

st.set_page_config(page_title="Real Estate Advisor", layout="wide")


st.title("🏡 Real Estate Investment Advisor")
st.write("Upload your dataset to analyze and get investment insights.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:

    # Load data
    df = load_data(uploaded_file)

    st.success("Dataset Loaded Successfully!")
    st.write("### First 5 Rows")
    st.dataframe(df.head())

    # Preprocess
    df = basic_cleaning(df)
    df = impute_missing(df)
    df = engineer_features(df)
    df = create_investment_label(df)
    df = predict_future_price(df)

    st.write("### ✔ Data Successfully Processed")

    # SECTION 1: Summary
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

    # SECTION 2: Price Distribution
    st.subheader("💰 Price Distribution")
    fig1 = px.histogram(df, x="Price_in_Lakhs", nbins=40, title="Distribution of Property Prices")
    st.plotly_chart(fig1, use_container_width=True)

    # SECTION 3: Investment Classification
    st.subheader("🏆 Investment Classification Breakdown")
    inv_counts = df["Predicted_Good_Investment"].value_counts().rename({0:"Not Good",1:"Good"})
    fig2 = px.pie(values=inv_counts.values, names=inv_counts.index, title="Good vs Not Good Investments")
    st.plotly_chart(fig2, use_container_width=True)

    # SECTION 4: Future Price Forecast
    st.subheader("📈 Future Price Forecast (5 Years)")
    fig3 = px.scatter(df, x="Price_in_Lakhs", y="Future_Price_in_Lakhs",
                      title="Current vs Future Price")
    st.plotly_chart(fig3, use_container_width=True)

    # SECTION 5: Download Cleaned File
    st.subheader("⬇ Download Processed Dataset")
    st.download_button(
        label="Download Cleaned CSV",
        data=df.to_csv(index=False),
        file_name="cleaned_real_estate_data.csv",
        mime="text/csv"
    )

else:
    st.info("Please upload a CSV file to continue.")

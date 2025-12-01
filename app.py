import streamlit as st
import pandas as pd
import plotly.express as px
from preprocessing import (
    load_data, basic_cleaning, impute_missing,
    engineer_features, create_investment_label, predict_future_price
)

st.set_page_config(page_title="Real Estate Investment Advisor", layout="wide")

st.title("🏠 Real Estate Investment Advisor")
st.write("Upload your dataset to analyze property insights!")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and process
    df = load_data(uploaded_file)
    df = basic_cleaning(df)
    df = impute_missing(df)
    df = engineer_features(df)
    df = create_investment_label(df)
    df = predict_future_price(df)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Dataset Preview",
        "📊 EDA",
        "⭐ Investment Score",
        "📈 Future Price Prediction"
    ])

    # --------------------------
    # TAB 1 — Dataset Preview
    # --------------------------
    with tab1:
        st.subheader("Cleaned Dataset")
        st.dataframe(df.head(100))

        st.download_button(
            "Download Cleaned Dataset",
            df.to_csv(index=False),
            "Cleaned_Real_Estate.csv",
            "text/csv"
        )

    # --------------------------
    # TAB 2 — EDA
    # --------------------------
    with tab2:
        st.subheader("Price Distribution")
        fig1 = px.histogram(df, x="Price_in_Lakhs", nbins=50, title="Distribution of Prices")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Price per SqFt by Property Type")
        fig2 = px.box(df, x="Property_Type", y="Price_per_SqFt", title="PPSF by Property Type")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Size vs Price Trend")
        fig3 = px.scatter(df, x="Size_in_SqFt", y="Price_in_Lakhs", opacity=0.2, trendline="ols")
        st.plotly_chart(fig3, use_container_width=True)

    # --------------------------
    # TAB 3 — Investment Score
    # --------------------------
    with tab3:
        st.subheader("Investment Recommendation")
        fig4 = px.pie(df, names="Predicted_Good_Investment",
                      title="Good Investment vs Not")
        st.plotly_chart(fig4, use_container_width=True)

        st.write("1 = Good Investment, 0 = Not Recommended")

    # --------------------------
    # TAB 4 — Future Price Prediction
    # --------------------------
    with tab4:
        st.subheader("Future Price after 5 Years (8% Annual Growth)")
        fig5 = px.scatter(df, x="Price_in_Lakhs", y="Future_Price_in_Lakhs", opacity=0.3)
        st.plotly_chart(fig5, use_container_width=True)

        st.write(df[["Price_in_Lakhs", "Future_Price_in_Lakhs"]].head())

else:
    st.info("Please upload a CSV file to begin.")

import pandas as pd
import numpy as np

# -------------------------------
# Load CSV (from upload)
# -------------------------------
def load_data(file):
    return pd.read_csv(file)

# -------------------------------
# Basic Cleaning
# -------------------------------
def basic_cleaning(df):
    df = df.rename(columns=lambda c: c.strip())

    if "ID" in df.columns:
        df = df.drop_duplicates(subset=["ID"])

    num_cols = [
        "BHK","Size_in_SqFt","Price_in_Lakhs","Price_per_SqFt","Year_Built",
        "Floor_No","Total_Floors","Age_of_Property","Nearby_Schools",
        "Nearby_Hospitals","Parking_Space"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = [
        "State","City","Locality","Property_Type","Furnished_Status","Security",
        "Amenities","Facing","Owner_Type","Availability_Status",
        "Public_Transport_Accessibility"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df

# -------------------------------
# Missing Value Imputation
# -------------------------------
def impute_missing(df):
    numeric = df.select_dtypes(include=[np.number]).columns
    for c in numeric:
        df[c] = df[c].fillna(df[c].median())

    categorical = df.select_dtypes(include="object").columns
    for c in categorical:
        df[c] = df[c].fillna("Unknown")

    return df

# -------------------------------
# Feature Engineering
# -------------------------------
def engineer_features(df, current_year=None):
    if current_year is None:
        current_year = pd.Timestamp.now().year

    # Compute missing PPSF
    if "Price_per_SqFt" in df.columns:
        mask = (df["Price_per_SqFt"].isna()) | (df["Price_per_SqFt"] <= 0)
        df.loc[mask, "Price_per_SqFt"] = (df["Price_in_Lakhs"] * 1e5) / df["Size_in_SqFt"]

    df["Price_per_SqFt_lakhs"] = df["Price_in_Lakhs"] / df["Size_in_SqFt"]

    df["Age_of_Property"] = current_year - df["Year_Built"]

    df["Amenities_count"] = df["Amenities"].apply(
        lambda s: len([x for x in str(s).split(",") if x.strip() != ""])
    )

    df["Schools_per_1000sqft"] = df["Nearby_Schools"] / (df["Size_in_SqFt"]/1000)

    mapping = {"High":3, "Medium":2, "Low":1, "Very High":4, "Unknown":0}
    df["PublicTransportScore"] = df["Public_Transport_Accessibility"].map(mapping).fillna(0)

    df["Is_Ready_to_Move"] = df["Availability_Status"].apply(
        lambda x: 1 if "ready" in x.lower() else 0
    )

    df["Has_Security"] = df["Security"].apply(
        lambda x: 1 if x.lower() not in ["no", "unknown"] else 0
    )

    return df

# -------------------------------
# Investment Label
# -------------------------------
def create_investment_label(df):
    city_med = df.groupby("City")["Price_per_SqFt"].transform("median")
    df["Cheap_vs_City"] = (df["Price_per_SqFt"] <= city_med).astype(int)

    overall_median = df["Price_in_Lakhs"].median()
    df["Cheap_vs_Median"] = (df["Price_in_Lakhs"] <= overall_median).astype(int)

    df["BHK_norm"] = (df["BHK"] - df["BHK"].min()) / (df["BHK"].max() - df["BHK"].min())
    df["Amenities_norm"] = (df["Amenities_count"] - df["Amenities_count"].min()) / (
        df["Amenities_count"].max() - df["Amenities_count"].min()
    )

    df["MultiFactorScore"] = (
        df["BHK_norm"]*0.25 +
        df["Is_Ready_to_Move"]*0.20 +
        df["Has_Security"]*0.15 +
        df["Amenities_norm"]*0.20 +
        (df["PublicTransportScore"]/3)*0.20
    )

    df["MultiFactorScore_pct"] = df["MultiFactorScore"] * 100

    df["Predicted_Good_Investment"] = (
        (df["Cheap_vs_City"] + 
         df["Cheap_vs_Median"] + 
         (df["MultiFactorScore_pct"] >= 60)) >= 2
    ).astype(int)

    return df

# -------------------------------
# Future Price Prediction
# -------------------------------
def predict_future_price(df, years=5, rate=0.08):
    df["Future_Price_in_Lakhs"] = df["Price_in_Lakhs"] * ((1+rate)**years)
    df["Future_Price_Change_Pct"] = (df["Future_Price_in_Lakhs"] / df["Price_in_Lakhs"] - 1) * 100
    return df

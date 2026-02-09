import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Video Game Sales Dashboard",
    layout="wide"
)

st.title("🎮 Video Game Sales Dashboard")
st.markdown(
    """
    **Purpose:**  
    Interactive exploration of video game sales trends by genre, platform, and region.
    """
)

@st.cache_data
def load_raw():
    return pd.read_csv("data/raw/vgsales.csv")

@st.cache_data
def load_genre_year():
    return pd.read_csv("data/processed/genre_year_sales.csv")

@st.cache_data
def load_region_trends():
    return pd.read_csv("data/processed/region_trends.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/ridge_model.pkl")

@st.cache_resource
def load_xgb_model():
    return joblib.load("models/xgb_model.pkl")

@st.cache_data
def load_model_features():
    df = pd.read_csv("data/processed/model_features.csv")

    # Remove target column if present
    feature_cols = [c for c in df.columns if c != "Global_Sales_M"]

    return feature_cols


df_raw = load_raw()
st.subheader("Raw Dataset Preview")
st.dataframe(df_raw.head())

tab1, tab2, tab3, tab4 = st.tabs([
    "Genre Trends",
    "Platform Lifecycles",
    "Regional Preferences",
    "Sales Prediction"
])

with tab1:
    st.subheader("Shooter vs Role-Playing Sales Trends")

    genre_year = load_genre_year()

    focus = genre_year[genre_year["Genre"].isin(["Shooter", "Role-Playing"])]

    fig, ax = plt.subplots(figsize=(10, 5))

    for genre in ["Shooter", "Role-Playing"]:
        subset = focus[focus["Genre"] == genre]
        ax.plot(
            subset["Year"],
            subset["Global_Sales_M"],
            marker="o",
            label=genre
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Global Sales (Millions)")
    ax.set_title("Global Sales Trends: Shooter vs Role-Playing")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

with tab2:
    st.subheader("PS3 vs PS4 Platform Lifecycles")

    lifecycle = pd.read_csv("data/processed/ps3_ps4_lifecycle.csv")

    # Normalize sales per platform (peak = 1)
    lifecycle["Normalized_Sales"] = (
        lifecycle
        .groupby("Platform")["Global_Sales_M"]
        .transform(lambda x: x / x.max())
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for platform in ["PS3", "PS4"]:
        subset = lifecycle[lifecycle["Platform"] == platform]
        ax.plot(
            subset["Year"],
            subset["Normalized_Sales"],
            marker="o",
            label=platform
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized Sales (Peak = 1)")
    ax.set_title("Normalized Platform Lifecycles")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

with tab3:
    st.subheader("Regional Preference for Role-Playing Games")

    region_df = load_region_trends()

    # Filter to Role-Playing only
    rpg = region_df[region_df["Genre"] == "Role-Playing"].copy()

    # Aggregate RPG sales by region (NO Other)
    rpg_region_sales = pd.DataFrame({
        "Region": ["NA", "EU", "JP"],
        "RPG_Sales_M": [
            rpg["NA_Sales_M"].sum(),
            rpg["EU_Sales_M"].sum(),
            rpg["JP_Sales_M"].sum()
        ]
    })

    # Total sales by region (all genres, from raw data)
    total_region_sales = pd.DataFrame({
        "Region": ["NA", "EU", "JP"],
        "Total_Sales_M": [
            df_raw["NA_Sales_M"].sum(),
            df_raw["EU_Sales_M"].sum(),
            df_raw["JP_Sales_M"].sum()
        ]
    })

    # Merge & compute preference index
    comparison = rpg_region_sales.merge(total_region_sales, on="Region")

    comparison["RPG_Preference_Index"] = (
        (comparison["RPG_Sales_M"] / comparison["RPG_Sales_M"].sum()) /
        (comparison["Total_Sales_M"] / comparison["Total_Sales_M"].sum())
    )

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.bar(
        comparison["Region"],
        comparison["RPG_Preference_Index"],
        color="steelblue"
    )

    ax.axhline(
        1,
        linestyle="--",
        color="red",
        label="Neutral Preference (Index = 1)"
    )

    ax.set_ylabel("RPG Preference Index")
    ax.set_title("Regional Preference for Role-Playing Games")
    ax.legend()

    for i, v in enumerate(comparison["RPG_Preference_Index"]):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center")

    st.pyplot(fig)

    st.caption(
        "Japan strongly over-indexes on Role-Playing games, indicating a cultural and market preference relative to its overall market size."
    )

with tab4:
    st.subheader("Global Sales Prediction (ML Model)")

    st.markdown(
        """
        This tool uses a **Ridge Regression model** trained on historical video game data
        to predict **Global Sales (in millions)** based on release year, platform,
        genre, and publisher strength.
        """
    )

    # Load model & feature schema
    model = load_model()
    feature_names = load_model_features()

    # --- User Inputs ---
    col1, col2 = st.columns(2)

    with col1:
        year = st.slider("Release Year", 1980, 2020, 2010)
        publisher_strength = st.slider(
            "Publisher Strength (0–1 scale)",
            min_value=0.0,
            max_value=1.0,
            value=0.5
        )

    with col2:
        platform = st.selectbox(
            "Platform",
            sorted(df_raw["Platform"].unique())
        )

        genre = st.selectbox(
            "Genre",
            sorted(df_raw["Genre"].unique())
        )

    # --- Build feature vector ---
    input_data = pd.DataFrame(0, index=[0], columns=feature_names)

    # Numeric features
    if "Year" in input_data.columns:
        input_data["Year"] = year

    if "Publisher_Strength" in input_data.columns:
        input_data["Publisher_Strength"] = publisher_strength

    # One-hot features
    platform_col = f"Platform_{platform}"
    genre_col = f"Genre_{genre}"

    if platform_col in input_data.columns:
        input_data[platform_col] = 1

    if genre_col in input_data.columns:
        input_data[genre_col] = 1

    # --- Prediction ---
    if st.button("Predict Global Sales"):
        log_prediction = model.predict(input_data)[0]

        # Convert from log scale to real sales (millions)
        prediction = max(0, np.expm1(log_prediction))

        st.metric(
            label="Predicted Global Sales",
            value=f"{prediction:.2f} million units"
        )

        st.caption(
            "Prediction is based on nonlinear historical sales patterns learned from the data. "
            "Values represent expected sales, not guarantees."
        )

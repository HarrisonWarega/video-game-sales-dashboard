import streamlit as st
import pandas as pd
import joblib

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
    st.info("Genre trends visuals")

with tab2:
    st.info("Platform lifecycle analysis")

with tab3:
    st.info("Regional preference charts ")

with tab4:
    st.info("Sales prediction model")

# 🎮 Video Game Sales Analytics Dashboard

## Overview
This project is an **interactive data analytics and machine learning dashboard** built with **Streamlit** to explore historical video game sales data.  
It analyzes **genre trends, platform lifecycles, regional preferences**, and includes a **machine learning model** to predict global sales.

The dashboard is designed to support **data-driven insights** into how video game markets evolve over time across genres, platforms, and regions.

---

## Dataset
The project is based on a cleaned and enriched version of the classic **Video Game Sales dataset**, containing:

- Game metadata (platform, genre, publisher, year)
- Regional sales (NA, EU, JP, Other)
- Engineered features (log-transformed sales, publisher strength, era labels)
- Aggregated and normalized datasets for visualization

Raw data is stored separately from processed analytical datasets to reflect real-world data pipelines.

---

## Project Structure
video-game-sales-dashboard/
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── data/
│ ├── raw/
│ │ └── vgsales.csv # Base cleaned dataset
│ └── processed/
│ ├── genre_year_sales.csv
│ ├── ps3_ps4_lifecycle.csv
│ ├── region_trends.csv
│ └── model_features.csv
│
├── models/
│ ├── ridge_model.pkl # Baseline regression model
│ └── xgb_model.pkl # Improved XGBoost model

---

## Dashboard Features (Primary Tasks)

### 1️⃣ Genre Sales Trends (Shooter vs Role-Playing)
- Line chart showing **global sales trends over time**
- Direct comparison between **Shooter** and **Role-Playing** genres
- Highlights long-term shifts in player demand

**Insight:**  
Shooters dominate peak console eras, while Role-Playing games show stronger persistence and regional specialization.

---

### 2️⃣ Platform Lifecycles (PS3 vs PS4)
- Normalized lifecycle curves (peak sales = 1)
- Visual comparison of adoption, peak, and decline phases
- Allows lifecycle comparison independent of absolute sales scale

**Insight:**  
PS3 shows a longer plateau, while PS4 exhibits a sharper growth–decline pattern, reflecting market acceleration in later console generations.

---

### 3️⃣ Regional Preference for Role-Playing Games
- RPG Preference Index by region
- Index > 1 indicates over-indexing relative to market size

**Insight:**  
Japan strongly over-indexes on Role-Playing games, confirming a cultural and market preference compared to North America and Europe.

---

### 4️⃣ Global Sales Prediction (Machine Learning)
- Interactive prediction tool using **XGBoost Regression**
- Inputs:
  - Release Year
  - Platform
  - Genre
  - Publisher Strength
- Outputs predicted **Global Sales (in millions)**

**Modeling Notes:**
- Target variable modeled in log-space to handle skewed sales distribution
- Predictions are transformed back to original scale
- Negative predictions are clipped to zero (business constraint)

---

## Technologies Used
- **Python**
- **Pandas / NumPy**
- **Matplotlib**
- **Scikit-learn**
- **XGBoost**
- **Streamlit**
- **Git & GitHub**

---

## How to Run Locally

1. Clone the repository
```bash
git clone <your-repo-url>
cd video-game-sales-dashboard

2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the dashboard
streamlit run app.py


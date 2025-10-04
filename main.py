import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Medical Cost Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #0f172a);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    font-family: 'Poppins', sans-serif;
    color: #e2e8f0;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar fix */
section[data-testid="stSidebar"] {
    background: #1e293b;
    color: white !important;
    border-right: 2px solid #334155;
    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    padding: 1rem;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span {
    color: white !important;
}
section[data-testid="stSidebar"] select,
section[data-testid="stSidebar"] option,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] div[role="listbox"] {
    color: white !important;
    background-color: #1e293b !important;
}
section[data-testid="stSidebar"] button {
    color: white !important;
    background-color: #334155 !important;
    border: none;
}

.header-section {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    padding: 2rem;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 2.5rem;
    box-shadow: 0 12px 40px rgba(0,0,0,0.5);
}
.header-title {
    font-size: 2.6rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    color: #f8fafc;
    text-shadow: 0 0 10px rgba(0,245,212,0.6);
}
.header-subtitle {
    font-size: 1.1rem;
    color: #cbd5e1;
}
.glass-card {
    background: rgba(30,41,59,0.65);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.5rem;
}
.profile-section {
    padding: 1.5rem;
    border-radius: 20px;
    background: rgba(51,65,85,0.7);
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    text-align: center;
}
.profile-avatar-container {
    width: 140px;
    height: 140px;
    background: linear-gradient(45deg, #a855f7, #6366f1, #06b6d4);
    background-size: 200% 200%;
    animation: gradientSpin 8s ease infinite;
    border-radius: 50%;
    display: flex;
    justify-content: center;
    align-items: center;
    margin: auto;
    margin-bottom: 1.5rem;
}
@keyframes gradientSpin {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.profile-avatar {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    border: 4px solid #f1f5f9;
}
.profile-item {
    font-size: 1rem;
    margin: 0.5rem 0;
    color: #f1f5f9;
}
.metric-card {
    background: rgba(23, 27, 45, 0.8);
    backdrop-filter: blur(10px);
    padding: 1.2rem;
    border-radius: 16px;
    text-align: center;
    font-weight: 600;
    color: white;
    box-shadow: 0 6px 20px rgba(0,0,0,0.5);
}
.metric-title {
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
    color: #94a3b8;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #00f5d4;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load Data & Train Model
# -------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv("insurance.csv")
    X = df.drop("charges", axis=1)
    y = df["charges"]

    categorical = ["sex", "smoker", "region"]
    numeric = ["age", "bmi", "children"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(), categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    model.fit(X, y)
    return model, df

model, df = train_model()

# -------------------------
# Sidebar Input
# -------------------------
st.sidebar.title("Patient Information")
age = st.sidebar.slider("Age", 18, 100, 30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
children = st.sidebar.slider("Children", 0, 5, 0)
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# -------------------------
# Header
# -------------------------
st.markdown("""
<div class="header-section">
    <div class="header-title">Medical Cost Predictor</div>
    <div class="header-subtitle">Estimate health insurance charges based on patient details</div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Predict Button (on main page)
# -------------------------
if st.button("Predict Cost"):
    # Prediction
    input_data = pd.DataFrame({
        "age": [age], "sex": [sex], "bmi": [bmi],
        "children": [children], "smoker": [smoker], "region": [region]
    })
    prediction = model.predict(input_data)[0]
    prediction_inr = prediction * 83  # approximate USD → INR

    # Layout
    col1, col2 = st.columns([1, 2])
    with col1:
        profile_img = "https://cdn-icons-png.flaticon.com/512/2922/2922510.png" if sex == "male" else "https://cdn-icons-png.flaticon.com/512/2922/2922561.png"
        st.markdown(f"""
        <div class="profile-section">
            <div class="profile-avatar-container">
                <img src="{profile_img}" class="profile-avatar">
            </div>
            <div class="profile-item">Sex: {sex.capitalize()}</div>
            <div class="profile-item">Age: {age}</div>
            <div class="profile-item">BMI: {bmi}</div>
            <div class="profile-item">Children: {children}</div>
            <div class="profile-item">Smoker: {smoker.capitalize()}</div>
            <div class="profile-item">Region: {region.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <h3 style="text-align:center; color:#f1f5f9;">Predicted Medical Cost</h3>
            <p class="metric-value" style="text-align:center;">${prediction:,.2f}</p>
            <p style="text-align:center; color:#94a3b8;">≈ ₹{prediction_inr:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.markdown(f"""<div class="metric-card"><div class="metric-title">Dataset Size</div><div class="metric-value">{len(df)}</div></div>""", unsafe_allow_html=True)
        m2.markdown(f"""<div class="metric-card"><div class="metric-title">Average Charges</div><div class="metric-value">${df['charges'].mean():,.0f}</div></div>""", unsafe_allow_html=True)
        m3.markdown(f"""<div class="metric-card"><div class="metric-title">Maximum Charges</div><div class="metric-value">${df['charges'].max():,.0f}</div></div>""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # -------------------------
        # Graphs (queue structure)
        # -------------------------
        # Graph 1: Charges Distribution
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(df["charges"], bins=30, alpha=0.8, color="#06b6d4", edgecolor="white")
        ax.axvline(prediction, color="#a855f7", linestyle="--", linewidth=2, label="Prediction")
        ax.set_title("Distribution of Medical Charges", color="white")
        ax.set_xlabel("Charges", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="black", edgecolor="white", labelcolor="white")
        fig.patch.set_alpha(0)
        ax.set_facecolor("black")
        st.pyplot(fig)

        # Graph 2: BMI vs Charges
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.scatter(df["bmi"], df["charges"], alpha=0.6, color="#22d3ee")
        ax2.scatter(bmi, prediction, color="red", s=100, label="You")
        ax2.set_title("BMI vs Charges", color="white")
        ax2.set_xlabel("BMI", color="white")
        ax2.set_ylabel("Charges", color="white")
        ax2.tick_params(colors="white")
        ax2.legend(facecolor="black", edgecolor="white", labelcolor="white")
        fig2.patch.set_alpha(0)
        ax2.set_facecolor("black")
        st.pyplot(fig2)

        # Graph 3: Age vs Charges
        fig3, ax3 = plt.subplots(figsize=(8, 3))
        ax3.scatter(df["age"], df["charges"], alpha=0.6, color="#f59e0b")
        ax3.scatter(age, prediction, color="red", s=100, label="You")
        ax3.set_title("Age vs Charges", color="white")
        ax3.set_xlabel("Age", color="white")
        ax3.set_ylabel("Charges", color="white")
        ax3.tick_params(colors="white")
        ax3.legend(facecolor="black", edgecolor="white", labelcolor="white")
        fig3.patch.set_alpha(0)
        ax3.set_facecolor("black")
        st.pyplot(fig3)

        # Graph 4: Smoker vs Charges
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        df.boxplot(column="charges", by="smoker", ax=ax4, patch_artist=True,
                   boxprops=dict(facecolor="#06b6d4", color="white"),
                   medianprops=dict(color="red"),
                   whiskerprops=dict(color="white"),
                   capprops=dict(color="white"))
        ax4.set_title("Smoker vs Charges", color="white")
        ax4.set_xlabel("Smoker", color="white")
        ax4.set_ylabel("Charges", color="white")
        ax4.tick_params(colors="white")
        fig4.suptitle('')
        fig4.patch.set_alpha(0)
        ax4.set_facecolor("black")
        st.pyplot(fig4)

        # Graph 5: Children vs Charges
        fig5, ax5 = plt.subplots(figsize=(8, 3))
        df.boxplot(column="charges", by="children", ax=ax5, patch_artist=True,
                   boxprops=dict(facecolor="#22d3ee", color="white"),
                   medianprops=dict(color="red"),
                   whiskerprops=dict(color="white"),
                   capprops=dict(color="white"))
        ax5.set_title("Children vs Charges", color="white")
        ax5.set_xlabel("Number of Children", color="white")
        ax5.set_ylabel("Charges", color="white")
        ax5.tick_params(colors="white")
        fig5.suptitle('')
        fig5.patch.set_alpha(0)
        ax5.set_facecolor("black")
        st.pyplot(fig5)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap.plots._waterfall")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import base64
import shap

# Page config
st.set_page_config(page_title="Employee Salary Predictor", layout="wide")

# Background image setup
background_image = "D:/Employee Salary Prediction/app/background.jpg"
if os.path.exists(background_image):
    with open(background_image, "rb") as f:
        encoded_bg = base64.b64encode(f.read()).decode()
    bg_img_style = f"""
    <style>
    .stApp {{
        background-image: url('data:image/jpg;base64,{encoded_bg}');
        background-size: cover;
        background-position: center;
    }}
    </style>
    """
    st.markdown(bg_img_style, unsafe_allow_html=True)

# Load model
model_path = "D:/Employee Salary Prediction/model/salary_model.pkl"
model = joblib.load(model_path)

# Load data for SHAP explainer
data_path = "D:/Employee Salary Prediction/data/salary_prediction_data.csv"
df_shap = pd.read_csv(data_path).drop('Salary', axis=1)

# SHAP explainer
explainer = shap.Explainer(
    model.named_steps['regressor'],
    model.named_steps['preprocess'].transform(df_shap)
)

# Streamlit title
st.title("Employee Salary Prediction App")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
education = st.sidebar.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
job_title = st.sidebar.selectbox("Job Title", ["Software Engineer", "Data Scientist", "Manager", "HR", "Analyst"])
location = st.sidebar.selectbox("Location", [
    "New York", "San Francisco", "London",
    "Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai", "Pune", "Kolkata"
])

input_dict = {
    'Age': age,
    'Experience': experience,
    'Gender': gender,
    'Education': education,
    'Job_Title': job_title,
    'Location': location
}

input_df = pd.DataFrame([input_dict])

# Track prediction state
if 'predicted' not in st.session_state:
    st.session_state.predicted = False

# Predict button
if st.sidebar.button("Predict Salary"):
    try:
        salary_pred = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ₹{salary_pred:,.2f}")

        st.subheader("Explain Prediction with SHAP")

        transformed_input = model.named_steps['preprocess'].transform(input_df)
        shap_values = explainer(transformed_input)

        st.write("Feature impact on prediction:")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.waterfall_plot(shap_values[0], max_display=10)
        st.pyplot(fig)

        st.session_state.predicted = True

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.session_state.predicted = False

# Show EDA if prediction not done
if not st.session_state.predicted:
    st.header("Exploratory Data Analysis")

    @st.cache_data
    def load_data():
        return pd.read_csv(data_path)

    df = load_data()

    fig1, ax1 = plt.subplots()
    ax1.hist(df['Salary'], bins=30, color='skyblue', edgecolor='black')
    ax1.set_title("Salary Distribution")
    ax1.set_xlabel("Salary")
    ax1.set_ylabel("Frequency")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="Job_Title", y="Salary", ax=ax2)
    ax2.set_title("Salary by Job Title")
    ax2.set_xticks(range(len(df['Job_Title'].unique())))
    ax2.set_xticklabels(df['Job_Title'].unique(), rotation=45, ha="right")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.violinplot(data=df, x="Education", y="Salary", ax=ax3)
    ax3.set_title("Salary by Education Level")
    ax3.set_xticks(range(len(df['Education'].unique())))
    ax3.set_xticklabels(df['Education'].unique(), rotation=20, ha="right")
    st.pyplot(fig3)

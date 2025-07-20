import pandas as pd
import joblib
import shap

# Load model pipeline
model = joblib.load(r"D:\Employee Salary Prediction\model\salary_model.pkl")

# Load dataset features only (without target)
df = pd.read_csv(r"D:\Employee Salary Prediction\data\salary_prediction_data.csv").drop('Salary', axis=1)

# Create SHAP TreeExplainer for the RandomForest regressor
explainer = shap.TreeExplainer(model.named_steps['regressor'])

def predict_salary(input_dict):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

def get_shap_values(input_dict):
    input_df = pd.DataFrame([input_dict])
    transformed_input = model.named_steps['preprocess'].transform(input_df)
    shap_values = explainer.shap_values(transformed_input)
    return shap_values

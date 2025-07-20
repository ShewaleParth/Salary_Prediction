import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load data
df = pd.read_csv(r"D:\Employee Salary Prediction\data\salary_prediction_data.csv")

# Clean data
df = df.dropna()

# Features and target
X = df.drop('Salary', axis=1)
y = df['Salary']

# Columns
numeric = ['Age', 'Experience']
categorical = ['Gender', 'Education', 'Job_Title', 'Location']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

# Model pipeline
model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
print("R² Score:", r2_score(y_test, preds))

# Save model
joblib.dump(model, r"D:\Employee Salary Prediction\model\salary_model.pkl")
print("Model saved successfully.")

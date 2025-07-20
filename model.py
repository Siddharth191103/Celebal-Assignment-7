import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and dataset
model = joblib.load("car_price_model.pkl")
df = pd.read_csv("cars.csv")
df = df.dropna()

st.set_page_config(page_title="Car Price Predictor", layout="wide")
st.title("🚗 Car Price Prediction App")

st.sidebar.header("Enter Car Features")

def user_input_features():
    input_data = {}
    for col in df.drop("Price", axis=1).columns:
        if df[col].dtype == 'object':
            input_data[col] = st.sidebar.selectbox(col, df[col].unique())
        else:
            input_data[col] = st.sidebar.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    return pd.DataFrame([input_data])

input_df = user_input_features()
prediction = model.predict(input_df)[0]

st.subheader("📈 Prediction Result")
st.success(f"Estimated Price: ₹ {prediction:,.2f}")

st.subheader("🧾 Input Features")
st.write(input_df)

st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

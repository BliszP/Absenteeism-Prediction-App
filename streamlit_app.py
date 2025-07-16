import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Absenteeism Predictor", layout="wide")

# === Custom App Header with Description ===
st.markdown("""
<style>
.header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #f9f9f9;
    padding: 1rem 2rem;
    border-radius: 0.5rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #4CAF50;
}
.left {
    text-align: left;
    font-size: 14px;
    color: #333;
}
.center {
    text-align: center;
    flex: 1;
    font-size: 20px;
    font-weight: bold;
    color: #2c3e50;
}
</style>
<div class="header-container">
  <div class="left">Contact: ibodjeb@gmail.com</div>
  <div class="center">
    Absenteeism Prediction App<br>
    Created by Blessing Ibodje
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
The **Absenteeism Prediction App** is a machine learning-powered solution designed to help organizations predict employee absence based on key behavioral, medical, and logistical factors.

By training on historical absenteeism data, the model helps HR and operations teams forecast workforce availability and plan ahead — reducing downtime, improving productivity, and enhancing resource allocation.

With this tool, any company can turn their internal absence records into a **predictive engine** tailored to their workforce dynamics.

💼 **Got workforce data?** I can help you transform it into a strategic prediction tool just like this — customized with your features, patterns, and HR priorities.
""")

st.markdown("""
### 🧾 Absence Categories
- **Reason_1** — Certain Diseases (e.g., infections, musculoskeletal conditions)
- **Reason_2** — Childbirth, Pregnancy
- **Reason_3** — Poisoning (e.g., food or chemical exposure)
- **Reason_4** — Routine Medical Check-Ups
""")

# Define expected feature columns
feature_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month_Values',
                 'Transportation Expense', 'Age', 'Body Mass Index',
                 'Education', 'Children', 'Pets']

# === Function to preprocess uploaded data ===
def preprocess_uploaded_data(df):
    df_with_predictions = df.copy()
    if 'ID' in df.columns:
        df = df.drop(['ID'], axis=1)
    df['Absenteeism Time in Hours'] = np.nan
    reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
    reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
    reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
    reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
    reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
    df = df.drop(['Reason for Absence'], axis=1)
    df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
    df.columns = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                  'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                  'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
    df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense',
             'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education',
             'Children', 'Pets', 'Absenteeism Time in Hours']]
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Month_Values'] = df['Date'].dt.month
    df['Day of the Week'] = df['Date'].dt.weekday
    df = df.drop(['Date'], axis=1)
    df = df[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month_Values', 'Day of the Week',
             'Transportation Expense', 'Distance to Work', 'Age',
             'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
             'Pets', 'Absenteeism Time in Hours']]
    df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
    df = df.fillna(0)
    df = df.drop(['Absenteeism Time in Hours', 'Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
    df = df[feature_order]
    return df, df_with_predictions

# === Main Interaction Section ===
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
data = None

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Data")
    st.dataframe(raw_df.head())
    df, df_with_predictions = preprocess_uploaded_data(raw_df)
    data = df
else:
    st.subheader("✍️ Manual Input")
    with st.form("manual_form"):
        Reason_1 = st.selectbox("Reason Type 1", [0, 1])
        Reason_2 = st.selectbox("Reason Type 2", [0, 1])
        Reason_3 = st.selectbox("Reason Type 3", [0, 1])
        Reason_4 = st.selectbox("Reason Type 4", [0, 1])
        Month_Values = st.slider("Month", 1, 12, 6)
        Transportation_Expense = st.number_input("Transportation Expense", 0, 1000, 100)
        Age = st.number_input("Age", 18, 65, 30)
        BMI = st.number_input("Body Mass Index", 10, 60, 25)
        Education = st.selectbox("Education Level (0 = High School, 1 = Higher)", [0, 1])
        Children = st.slider("Number of Children", 0, 5, 1)
        Pets = st.slider("Number of Pets", 0, 5, 1)
        submit = st.form_submit_button("Predict")
    if submit:
        manual_input = pd.DataFrame([[Reason_1, Reason_2, Reason_3, Reason_4, Month_Values,
                                      Transportation_Expense, Age, BMI,
                                      Education, Children, Pets]], columns=feature_order)
        data = manual_input
        df_with_predictions = manual_input.copy()

if data is not None:
    scaled_inputs = scaler.transform(data)
    predictions = model.predict(scaled_inputs)
    probabilities = model.predict_proba(scaled_inputs)[:, 1]
    df_with_predictions['Prediction'] = predictions
    df_with_predictions['Probability'] = probabilities
    st.subheader("📊 Prediction Results")
    st.dataframe(df_with_predictions)
    st.success("✅ Prediction complete: 1 = likely absent, 0 = unlikely.")

    # === Interactive Dashboard ===
    st.markdown("---")
    st.subheader("📈 Interactive Dashboard")
    col1, col2 = st.columns(2)

    with col1:
        feature = st.selectbox("Select feature to plot by Age", feature_order, key="plot1")
        fig1 = px.scatter(df_with_predictions, x="Age", y=feature, color="Prediction", title="Age vs Selected Feature")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        reason_col = st.selectbox("Select Reason Column", ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4'], key="plot2")
        fig2 = px.bar(df_with_predictions, x=reason_col, y="Probability", color="Prediction",
                     title=f"{reason_col} vs Absenteeism Probability")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🚗 Transportation Expense Bubble Chart")
    fig3 = px.scatter(df_with_predictions, x="Transportation Expense", y="Probability", 
                      size="Children", color="Prediction", 
                      title="Transportation Expense vs Absence Probability (Bubble = Children)")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Please upload a CSV file or enter data manually to begin.")

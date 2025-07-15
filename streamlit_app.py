import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Absenteeism Prediction App")
st.markdown("Upload your CSV file to preprocess the data and predict absenteeism using a trained logistic regression model.")

# Upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.write(df.head())

    # ----------------- Data Preprocessing -------------------

    # Make a copy for final predictions
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
    df = df.drop(['Absenteeism Time in Hours'], axis=1)
    df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)

    feature_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month_Values',
                     'Transportation Expense', 'Age', 'Body Mass Index',
                     'Education', 'Children', 'Pets']

    df = df[feature_order]

    # Scale features
    scaled_inputs = scaler.transform(df)

    # Predict
    predictions = model.predict(scaled_inputs)
    probabilities = model.predict_proba(scaled_inputs)[:, 1]

    df_with_predictions['Prediction'] = predictions
    df_with_predictions['Probability'] = probabilities

    # ------------------- Output --------------------
    st.subheader("Predictions")
    st.write(df_with_predictions[['Prediction', 'Probability']].head())

    # ------------------- Visualizations --------------------
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_with_predictions['Age'], kde=True, bins=10, ax=ax1)
        ax1.set_title("Age Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df_with_predictions, x='Prediction', y='Transportation Expense', ax=ax2)
        ax2.set_title("Transportation Expense vs Prediction")
        st.pyplot(fig2)

    st.markdown("🎯 Logistic Regression model trained with 80-20 split. Feature-scaled. Prediction = 1 means **likely to be absent**.")

else:
    st.info("📂 Upload a CSV file to begin.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Absenteeism Predictor", layout="wide")
st.title("🧾 Absenteeism Prediction App")
st.markdown("Upload a CSV file **or** enter data manually to predict employee absenteeism and visualize patterns.")

# Define expected feature columns (after processing)
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
    st.markdown("---")
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
        Pet = st.slider("Number of Pets", 0, 5, 1)
        submit = st.form_submit_button("Predict")

    if submit:
        manual_input = pd.DataFrame([[Reason_1, Reason_2, Reason_3, Reason_4, Month_Values,
                                      Transportation_Expense, Age, BMI,
                                      Education, Children, Pets]], columns=feature_order)
        data = manual_input
        df_with_predictions = manual_input.copy()


# === Prediction and Output ===
if data is not None:
    scaled_inputs = scaler.transform(data)
    predictions = model.predict(scaled_inputs)
    probabilities = model.predict_proba(scaled_inputs)[:, 1]

    df_with_predictions['Prediction'] = predictions
    df_with_predictions['Probability'] = probabilities

    st.subheader("📊 Prediction Results")
    st.write(df_with_predictions)

    st.success("✅ Prediction complete: 1 = likely absent, 0 = unlikely.")

    # === Visualizations ===
    st.subheader("📈 Visualizations")

    with st.expander("🔎 Explore Features"):
        col1, col2 = st.columns(2)

        with col1:
            selected_feature = st.selectbox("Select feature for histogram", feature_order)
            fig1, ax1 = plt.subplots()
            sns.histplot(df_with_predictions[selected_feature], kde=True, ax=ax1)
            ax1.set_title(f"{selected_feature} Distribution")
            st.pyplot(fig1)

        with col2:
            y_feature = st.selectbox("Y-axis for boxplot", feature_order)
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df_with_predictions['Prediction'], y=df_with_predictions[y_feature], ax=ax2)
            ax2.set_title(f"{y_feature} vs Prediction")
            st.pyplot(fig2)

        # Correlation Heatmap
        st.subheader("🔗 Correlation Matrix")
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_with_predictions[feature_order].corr(), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)

else:
    st.info("Please upload a CSV file or enter data manually to begin.")

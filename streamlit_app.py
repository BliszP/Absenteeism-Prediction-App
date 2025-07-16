import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Absenteeism Prediction App", layout="wide")

# === Custom Header ===
st.markdown("""
<style>
body {
    background-color: #f4f6f8;
}
main .block-container {
    background-color: #f4f6f8 !important;
    padding-top: 1rem;
}

@media (prefers-color-scheme: dark) {
  .header, .subheader-left {
    color: #FFFFFF;
  }
}
@media (prefers-color-scheme: light) {
  .header {
    color: #2c3e50;
  }
  .subheader-left {
    color: #333333;
  }
}
.header {
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    margin-bottom: 0.2em;
}
.subheader-left {
    text-align: left;
    font-size: 14px;
    margin-bottom: 1.2em;
}
body {
    background-color: #f4f6f8;
}
</style>
<div class="header" style="background-color:#001f3f !impo"  // Glossy navy blue look
    }
  ]
} padding:10px; border-radius:8px;">
    <strong style="font-size: 24px; color: white; text-transform: uppercase;">Absenteeism Prediction App</strong>
</div>
<div class="subheader-left">
    <strong>Created by Blessing Ibodje</strong>  |
    <a href="mailto:ibodjeb@gmail.com" style="color: #66ccff; text-decoration: none;">Contact: ibodjeb@gmail.com</a>
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

# Feature columns
feature_order = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month_Values',
                 'Transportation Expense', 'Age', 'Body Mass Index',
                 'Education', 'Children', 'Pets']

# Preprocess uploaded CSV
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
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Month_Values'] = df['Date'].dt.month
    df = df.drop(['Date', 'Absenteeism Time in Hours', 'Distance to Work', 'Daily Work Load Average'], axis=1)
    df['Education'] = df['Education'].map({1: 0, 2: 1, 3: 1, 4: 1})
    df = df.fillna(0)
    df = df[feature_order]
    return df, df_with_predictions

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
data = None

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Data Preview")
    st.dataframe(raw_df.head())
    df, df_with_predictions = preprocess_uploaded_data(raw_df)
    data = df
else:
    with st.expander("✍️ Click to Enter Manual Input"):
        with st.form("manual_form"):
            Reason_1 = st.selectbox("Reason Type 1", [0, 1])
            Reason_2 = st.selectbox("Reason Type 2", [0, 1])
            Reason_3 = st.selectbox("Reason Type 3", [0, 1])
            Reason_4 = st.selectbox("Reason Type 4", [0, 1])
            Month_Values = st.slider("Month", 1, 12, 6)
            Transportation_Expense = st.number_input("Transportation Expense", 0, 1000, 100)
            Age = st.number_input("Age", 18, 65, 30)
            BMI = st.number_input("Body Mass Index", 10, 60, 25)
            Education = st.selectbox("Education Level", [0, 1])
            Children = st.slider("Children", 0, 5, 1)
            Pets = st.slider("Pets", 0, 5, 1)
            submit = st.form_submit_button("Predict")
        if submit:
            manual_input = pd.DataFrame([[Reason_1, Reason_2, Reason_3, Reason_4, Month_Values,
                                          Transportation_Expense, Age, BMI,
                                          Education, Children, Pets]], columns=feature_order)
            data = manual_input
            df_with_predictions = manual_input.copy()

# Prediction + Download
if data is not None:
    scaled = scaler.transform(data)
    preds = model.predict(scaled)
    probas = model.predict_proba(scaled)[:, 1]
    df_with_predictions['Prediction'] = preds
    df_with_predictions['Probability'] = probas
    st.subheader("📊 Prediction Results")
    st.dataframe(df_with_predictions)
    st.download_button("📥 Download Predictions as CSV", df_with_predictions.to_csv(index=False), file_name="predictions.csv")

    # Visual Insights Section Moved Below Prediction Output
    st.subheader("📊 Visual Insights from Analysis")
    image_paths = [
        "Age vs Probability.png",
        "Reasons vs Probability.png",
        "Transportation Expenses & Children.png"
    ]
    image_captions = [
        "Age vs Absence Probability",
        "Reasons vs Absence Probability",
        "Transportation Expenses & Children"
    ]
    for path, caption in zip(image_paths, image_captions):
        if os.path.exists(path):
            st.image(path, caption=caption, use_container_width=True)
        else:
            st.warning(f"⚠️ {caption} image not found. Please upload '{path}' to the app folder.")

else:
    st.info("Upload a CSV file or open manual input to get started.")

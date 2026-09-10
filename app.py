import streamlit as st
import pandas as pd
import joblib


# =========================
# LOAD MODEL AND FILES
# =========================

model = joblib.load('LogisticRegression_heart.pkl')
expected_columns = joblib.load('columns_heart.pkl')
scaler = joblib.load('scaler_heart.pkl')


# =========================
# PAGE TITLE
# =========================
st.markdown("""
<style>
    .stApp {
        background-color: grey;
        color: lightgreen;
    }
</style>
""", unsafe_allow_html=True)

st.title('Heart Disease Prediction by Hamid')

st.markdown('### Provide the following details')


# =========================
# USER INPUTS
# =========================

age = st.slider(
    'Age',
    18,
    100,
    40
)


sex = st.selectbox(
    'SEX',
    ['M', 'F']
)


chest_pain = st.selectbox(
    'Chest Pain Type',
    ['ATA', 'NAP', 'TA', 'ASY']
)


resting_bp = st.number_input(
    'Resting Blood Pressure',
    min_value=80,
    max_value=200,
    value=120
)


cholesterol = st.number_input(
    'Cholesterol',
    min_value=100,
    max_value=600,
    value=200
)


fasting_bs = st.selectbox(
    'Fasting BS',
    [0, 1]
)


resting_ecg = st.selectbox(
    'Resting ECG',
    ['Normal', 'ST', 'LVM']
)


max_hr = st.slider(
    'Max Heart Rate',
    60,
    220,
    150
)


exercise_angina = st.selectbox(
    'Exercise Induced Angina',
    ['Y', 'N']
)


oldpeak = st.slider(
    'OldPeak (ST Depression)',
    0.0,
    6.0,
    1.0
)


st_slope = st.selectbox(
    'ST Slope',
    ['UP', 'Flat', 'Down']
)


# =========================
# PREDICTION BUTTON
# =========================

if st.button('Predict'):

    # -------------------------
    # Create raw input
    # -------------------------

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }


    # -------------------------
    # Convert to DataFrame
    # -------------------------

    input_df = pd.DataFrame([raw_input])


    # -------------------------
    # Add missing columns
    # -------------------------

    for col in expected_columns:

        if col not in input_df.columns:
            input_df[col] = 0


    # -------------------------
    # Arrange columns
    # -------------------------

    input_df = input_df[expected_columns]


    # -------------------------
    # Scale input
    # -------------------------

    scaled_input = scaler.transform(input_df)


    # -------------------------
    # Prediction
    # -------------------------

    prediction = model.predict(scaled_input)[0]


    # -------------------------
    # Display result
    # -------------------------

    if prediction == 1:

        st.error('⚠️ High Risk of Heart Disease')

    else:

        st.success('✅ Low Risk of Heart Disease')
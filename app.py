
import streamlit as st
import pandas as pd
import joblib

# Load artifacts
rf_model = joblib.load("random_forest_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
feature_names = joblib.load("feature_names.pkl")
input_metadata = joblib.load("input_metadata.pkl")

# Extract metadata
raw_feature_columns = input_metadata["raw_feature_columns"]
numeric_features = input_metadata["numeric_features"]
categorical_features = input_metadata["categorical_features"]
categorical_options = input_metadata["categorical_options"]

st.title("Reading Preference Predictor")
st.write("Masukkan informasi berikut untuk memprediksi preferensi membaca Anda.")

# Create input widgets dynamically
user_input = {}

for col in raw_feature_columns:
    if col in numeric_features:
        # Numerical input
        min_val = 0 # You might want to get min/max from your training data for better bounds
        max_val = 100 # Adjust as needed
        if col == 'age':
            min_val = 17
            max_val = 45
        elif col == 'books_per_year':
            min_val = 0
            max_val = 30
        elif col in ['literary_self_rating', 'story_importance', 'writing_style_importance', 'deeper_meaning_preference', 'real_events_preference', 'easy_reading_preference', 'intellectual_challenge']:
            min_val = 1
            max_val = 5

        user_input[col] = st.number_input(
            f"Masukkan {col.replace('_', ' ').title()}:",
            min_value=min_val,
            max_value=max_val,
            value=int(max_val/2), # Default value
            key=f"input_{col}"
        )
    elif col in categorical_features:
        # Categorical input (dropdown)
        options = categorical_options[col]
        user_input[col] = st.selectbox(
            f"Pilih {col.replace('_', ' ').title()}:",
            options=options,
            key=f"input_{col}"
        )

if st.button("Prediksi Preferensi Membaca"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Preprocess the input using the loaded preprocessor
    processed_input = preprocessor.transform(input_df)
    processed_input_df = pd.DataFrame(processed_input, columns=feature_names)

    # Make prediction
    prediction = rf_model.predict(processed_input_df)
    prediction_proba = rf_model.predict_proba(processed_input_df)

    st.subheader("Hasil Prediksi:")
    st.success(f"Preferensi Membaca Anda kemungkinan besar adalah: **{prediction[0]}**")

    proba_df = pd.DataFrame({
        'Preference': rf_model.classes_,
        'Probability': prediction_proba[0]
    }).sort_values(by='Probability', ascending=False)

    st.write("Probabilitas untuk setiap preferensi:")
    st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))


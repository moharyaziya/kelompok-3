
import streamlit as st
import pandas as pd
import joblib

# Set page config for better aesthetics
st.set_page_config(
    page_title="Prediktor Preferensi Membaca Humaniora Sastra",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="auto"
)

# Load artifacts
# Using st.cache_resource to cache the model loading for better performance
@st.cache_resource
def load_artifacts():
    rf_model = joblib.load("random_forest_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    feature_names = joblib.load("feature_names.pkl")
    input_metadata = joblib.load("input_metadata.pkl")
    return rf_model, preprocessor, feature_names, input_metadata

rf_model, preprocessor, feature_names, input_metadata = load_artifacts()

# Extract metadata
raw_feature_columns = input_metadata["raw_feature_columns"]
numeric_features = input_metadata["numeric_features"]
categorical_features = input_metadata["categorical_features"]
categorical_options = input_metadata["categorical_options"]

# Helper function for translating column names to Indonesian
def get_translated_label(col_name):
    translation_map = {
        'age': 'Usia',
        'books_per_year': 'Jumlah Buku Per Tahun',
        'literary_self_rating': 'Peringkat Diri dalam Sastra',
        'story_importance': 'Pentingnya Alur Cerita',
        'writing_style_importance': 'Pentingnya Gaya Penulisan',
        'deeper_meaning_preference': 'Preferensi Makna Mendalam',
        'real_events_preference': 'Preferensi Kisah Nyata',
        'easy_reading_preference': 'Preferensi Bacaan Ringan',
        'intellectual_challenge': 'Tantangan Intelektual',
        'gender': 'Jenis Kelamin',
        'education': 'Pendidikan',
        'reading_frequency': 'Frekuensi Membaca'
    }
    return translation_map.get(col_name, col_name.replace('_', ' ').title())

st.title("📚 Prediktor Preferensi Membaca Humaniora Sastra")
st.markdown("""
Aplikasi ini memprediksi preferensi membaca (Fiksi, Non-fiksi, atau Keduanya) berdasarkan beberapa karakteristik pribadi dan preferensi membaca Anda.
Silakan isi informasi di bawah ini untuk mendapatkan prediksi Anda.
""")

# Input widgets
user_input = {}

st.header("Informasi Demografi & Kebiasaan Membaca")
col1, col2, col3 = st.columns(3) # Use columns for better layout

with col1:
    col = 'age'
    min_val = 17
    max_val = 45
    user_input[col] = st.number_input(
        f"{get_translated_label(col)}:",
        min_value=min_val,
        max_value=max_val,
        value=int(max_val/2),
        key=f"input_{col}"
    )

with col2:
    col = 'books_per_year'
    min_val = 0
    max_val = 30
    user_input[col] = st.number_input(
        f"{get_translated_label(col)}:",
        min_value=min_val,
        max_value=max_val,
        value=int(max_val/2),
        key=f"input_{col}"
    )

with col3:
    col = 'gender'
    options = categorical_options[col]
    user_input[col] = st.selectbox(
        f"{get_translated_label(col)}:",
        options=options,
        key=f"input_{col}"
    )

col1_cat, col2_cat = st.columns(2)
with col1_cat:
    col = 'education'
    options = categorical_options[col]
    user_input[col] = st.selectbox(
        f"{get_translated_label(col)}:",
        options=options,
        key=f"input_{col}"
    )
with col2_cat:
    col = 'reading_frequency'
    options = categorical_options[col]
    user_input[col] = st.selectbox(
        f"{get_translated_label(col)}:",
        options=options,
        key=f"input_{col}"
    )

st.header("Preferensi & Gaya Membaca")
st.markdown("Skala 1 (Sangat Tidak Penting/Setuju) hingga 5 (Sangat Penting/Setuju), kecuali 'Peringkat Diri Sastra' (1-7).")

rating_cols = [
    'literary_self_rating',
    'story_importance',
    'writing_style_importance',
    'deeper_meaning_preference',
    'real_events_preference',
    'easy_reading_preference',
    'intellectual_challenge'
]

# Use expander for aesthetic grouping of rating inputs
with st.expander("Isi Preferensi Membaca Anda"):
    for i in range(0, len(rating_cols), 2):
        cols = st.columns(2);
        for j in range(2):
            if i + j < len(rating_cols):
                col_name = rating_cols[i+j]
                with cols[j]:
                    min_val = 1
                    max_val = 5
                    if col_name == 'literary_self_rating':
                        max_val = 7 # This is the specific one that goes up to 7
                    user_input[col_name] = st.slider(
                        f"{get_translated_label(col_name)}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=int((min_val + max_val) / 2),
                        key=f"input_{col_name}"
                    )

st.markdown("---") # Separator

if st.button("Prediksi Preferensi Membaca Anda", type="primary", use_container_width=True):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Preprocess the input using the loaded preprocessor
    processed_input = preprocessor.transform(input_df)
    processed_input_df = pd.DataFrame(processed_input, columns=feature_names)

    # Make prediction
    prediction = rf_model.predict(processed_input_df)
    prediction_proba = rf_model.predict_proba(processed_input_df)

    st.subheader("🎉 Hasil Prediksi:")
    st.success(f"Preferensi Membaca Anda kemungkinan besar adalah: **{prediction[0]}**")

    st.markdown("---")
    st.subheader("Distribusi Probabilitas:")
    proba_df = pd.DataFrame({
        'Preferensi': rf_model.classes_,
        'Probabilitas': prediction_proba[0]
    }).sort_values(by='Probabilitas', ascending=False)

    # Display probabilities in a nice format
    st.dataframe(proba_df.style.format({'Probabilitas': '{:.2%}'}), hide_index=True)

    st.info("Prediksi ini didasarkan pada model Machine Learning yang dilatih dengan data survei. Hasil dapat bervariasi.")

st.markdown("---")
st.markdown("Dibuat dengan ❤️ untuk Humaniora Sastra.")

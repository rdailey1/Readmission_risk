from preprocessing import preprocess_data

processed_df = preprocess_data(
    raw_path="data/raw/diabetic_data.csv",
    output_path="data/processed/preprocessed_diabetes.csv"
)

print("Preprocessing complete.")
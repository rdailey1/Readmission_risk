import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load raw dataset and convert csv to dataframe to use inbuilt operations
df = pd.read_csv("data/raw/diabetic_data.csv")

print("Raw number of records:cols", df.shape)
# Drop ID cols and cols with majority missing content
columns_to_drop = [
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult"
]
df = df.drop(columns=columns_to_drop)
print("records:cols post removal of sparse and low-information features", df.shape)

# NaN valid option (effectively serves as unknown regardless of data type) but is
# not usable for ML purposes and may crash in the training process
# using 'unknown' should avoid this
df = df.replace("?", "unknown")

# Convert target variable 'readmitted' to binary. 1 if yes <30 days, else 0 for no
df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

# Separate target/features before encoding/scaling
target = df["readmitted"]
features = df.drop("readmitted", axis=1)

print("separated features/target")

# one-hot encoding takes categorical values such as race or non-binary vaules such as
# age and internally produces columns to numerically represent these features, one 
# col per category. required since AI/ML models cannot use strings, etc for math ops
print("start one-hot encoding")
features = pd.get_dummies(features, drop_first=True)
print("finish one-hot encoding")

# Scale numeric features to reduce bias large vals have over small vals in computation
print("begin scaling")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
print("end scaling")

print("populate new dataframe")
# create a new dataframe obj and populate it with scaled features and reassign col titles
processed_df = pd.DataFrame(features_scaled, columns=features.columns)

print("add target into new dataframe")
# add target col into the new scaled dataframe obj
processed_df["readmitted"] = target.values

print("saving preprocessed dataset")
# Save preprocessed dataset
processed_df.to_csv("data/processed/preprocessed_diabetes.csv", index=False)
print("completed saving")

print("\nPreprocessed dataset shape:", processed_df.shape)
print("Preprocessed dataset saved to data/processed/preprocessed_diabetes.csv")
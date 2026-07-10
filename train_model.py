import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==============================
# Load Dataset
# ==============================

df = pd.read_excel("Dumy_dataset.xlsx")

print("Dataset Loaded Successfully!")
print(df.head())

# ==============================
# Data Cleaning
# ==============================

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ==============================
# Select Features
# ==============================

features = [
    "Annual_Spending",
    "Orders_Count"
]

X = df[features]

# ==============================
# Feature Scaling
# ==============================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==============================
# Train KMeans
# ==============================

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

kmeans.fit(X_scaled)

# ==============================
# Save Model
# ==============================

joblib.dump(kmeans, "flipkart_kmeans.pkl")
joblib.dump(scaler, "flipkart_scaler.pkl")

print("Model Saved Successfully!")
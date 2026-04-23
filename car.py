import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

df = pd.read_csv("car_price_dataset.csv")

df = df.drop(columns=["Car_ID"])

le_brand = LabelEncoder()
le_fuel = LabelEncoder()
le_trans = LabelEncoder()

df["Brand"] = le_brand.fit_transform(df["Brand"])
df["Fuel_Type"] = le_fuel.fit_transform(df["Fuel_Type"])
df["Transmission"] = le_trans.fit_transform(df["Transmission"])

X = df.drop(columns=["Price"])
y = df["Price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_brand, "le_brand.pkl")
joblib.dump(le_fuel, "le_fuel.pkl")
joblib.dump(le_trans, "le_trans.pkl")

print("Model saved successfully")
import joblib
import pandas as pd


model = joblib.load("water_quality_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_water_quality(pH, turbidity, oxygen, conductivity):
    sample = pd.DataFrame(
    [[pH, turbidity, oxygen, conductivity]],
    columns=["pH", "Turbidity", "Dissolved_Oxygen", "Conductivity"]
)
    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    if prediction == 0:
        return "SAFE: Water quality is good."
    elif prediction == 1:
        return "MODERATE RISK: Water needs monitoring."
    else:
        return "UNSAFE: Immediate action required!"

if __name__ == "__main__":
    print("Water Quality Prediction System")
    pH = float(input("Enter pH value: "))
    turbidity = float(input("Enter Turbidity: "))
    oxygen = float(input("Enter Dissolved Oxygen: "))
    conductivity = float(input("Enter Conductivity: "))

    result = predict_water_quality(pH, turbidity, oxygen, conductivity)
    print("\nResult:", result)

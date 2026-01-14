import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "pH": np.random.uniform(5.5, 9.5, 500),
    "Turbidity": np.random.uniform(0.5, 10, 500),
    "Dissolved_Oxygen": np.random.uniform(3, 12, 500),
    "Conductivity": np.random.uniform(50, 1500, 500),
    "Contamination_Level": np.random.choice([0, 1, 2], 500)
}

df = pd.DataFrame(data)
df.to_csv("water_quality_data.csv", index=False)

print("Dataset saved as water_quality_data.csv")

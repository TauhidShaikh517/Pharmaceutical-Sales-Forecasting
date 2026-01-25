import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# =====================================================
# Step 1: Load Dataset
# =====================================================
data = pd.read_excel(
    r"D:\pythonProject\project\Pharmaceutical_Ecommerce_Sales_Data (1).xlsx",
    engine="openpyxl"
)

# Ensure Date column is datetime
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)

# =====================================================
# Step 2: Data Preprocessing
# =====================================================
# Rename Units_Sold → Sales_Value
data = data.rename(columns={"Units_Sold": "Sales_Value"})

# Ensure column exists and is numeric
data["Sales_Value"] = pd.to_numeric(data["Sales_Value"], errors="coerce")

# Interpolate missing sales values
data["Sales_Value"] = data["Sales_Value"].interpolate(method="linear")

# Handle zero or negative sales
data.loc[data["Sales_Value"] <= 0, "Sales_Value"] = np.nan

# Fill remaining NaNs with rolling mean
data["Sales_Value"] = data["Sales_Value"].fillna(
    data["Sales_Value"].rolling(3, min_periods=1).mean()
)

# Feature Engineering
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year
data["Quarter"] = data["Date"].dt.quarter

# Seasonality encoding
data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)

# Lag features
for lag in [1, 2, 3]:
    data[f"Sales_Lag_{lag}"] = data["Sales_Value"].shift(lag)

# Fill lag NaNs with backward fill
data = data.bfill()

# Outlier Treatment (IQR)
Q1 = data["Sales_Value"].quantile(0.25)
Q3 = data["Sales_Value"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data["Sales_Value"] = np.clip(data["Sales_Value"], lower_bound, upper_bound)

# =====================================================
# Step 3: GM(1,1) Model
# =====================================================
def gm11(x0):
    x0 = np.array(x0, dtype=float)
    x1 = np.cumsum(x0)
    z1 = (x1[:-1] + x1[1:]) / 2.0
    B = np.column_stack((-z1, np.ones(len(z1))))
    Y = x0[1:]
    a, b = np.linalg.lstsq(B, Y, rcond=None)[0]

    # Forecast function
    def forecast(n):
        return [
            x0[0] * np.exp(-a * k) + (b / a) * (1 - np.exp(-a * k))
            for k in range(n)
        ]

    return forecast(len(x0))

data["GM11_Forecast"] = gm11(data["Sales_Value"].values)

# =====================================================
# Step 4: SVR Forecasting
# =====================================================
data["TimeIndex"] = np.arange(len(data))

X = data[["TimeIndex"]].values
y = data["Sales_Value"].values

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

svr_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_model.fit(X_scaled, y_scaled)

# Forecast next 6 months
future_index = np.arange(len(data), len(data) + 6).reshape(-1, 1)
future_index_scaled = scaler_X.transform(future_index)
future_pred_scaled = svr_model.predict(future_index_scaled)
future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).ravel()

# =====================================================
# Step 5: Visualization
# =====================================================
plt.figure(figsize=(12, 6))

# Actual sales
plt.plot(data["Date"], data["Sales_Value"], label="Actual Sales")

# GM(1,1) forecast
plt.plot(data["Date"], data["GM11_Forecast"], label="GM(1,1) Forecast")

# SVR in-sample forecast
svr_in_sample = scaler_y.inverse_transform(
    svr_model.predict(X_scaled).reshape(-1, 1)
)
plt.plot(data["Date"], svr_in_sample, label="SVR (In-Sample)")

# SVR future forecast
future_dates = pd.date_range(
    start=data["Date"].iloc[-1] + pd.DateOffset(months=1),
    periods=6,
    freq="ME"  # ✅ Month End, uppercase required
)
plt.plot(future_dates, future_pred, "--", label="SVR Future Forecast")

plt.legend()
plt.title("Pharmaceutical E-Commerce Sales Forecasting")
plt.xlabel("Date")
plt.ylabel("Sales Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# Step 6: Save Results
# =====================================================
data.to_csv("forecast_results.csv", index=False)
print("📁 Forecast results saved to forecast_results.csv")


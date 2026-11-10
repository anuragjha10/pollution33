import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset (India air quality data, one reading per station per pollutant)
df = pd.read_csv("air_quality.csv")
print("Dataset shape:", df.shape)
print(df.head())

# 2. Understand the data
print("\nColumn types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nPollutant types:\n", df["pollutant_id"].value_counts())

# 3. Clean the data
# country and last_update have the same value in every row, so they give no information
df = df.drop(columns=["country", "last_update"])

# "NA" in the file means the station sent no reading, so remove those rows
df = df.dropna(subset=["pollutant_avg"])
df = df.drop_duplicates()
print("\nData after cleaning:", df.shape)

# keep a copy for the charts (before converting pollutant_id to 0/1 columns)
original = df.copy()

# 4. Convert the text column pollutant_id (CO, NO2, PM2.5 ...) into 0/1 columns
df = pd.get_dummies(df, columns=["pollutant_id"], dtype=int)

# 5. Select input columns and target column
# Inputs: where the station is + which pollutant it measures
# Target: the average pollutant level
pollutant_cols = [c for c in df.columns if c.startswith("pollutant_id_")]
X = df[["latitude", "longitude"] + pollutant_cols]
y = df["pollutant_avg"]

# 6. Simple data visualizations
plt.figure(figsize=(7, 4))
plt.hist(y, bins=30, edgecolor="black")
plt.title("Distribution of Average Pollutant Level")
plt.xlabel("Pollutant Average")
plt.tight_layout()
plt.savefig("fig1_distribution.png", dpi=300)
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=original, x="pollutant_id", y="pollutant_avg")
plt.title("Average Level of Each Pollutant")
plt.tight_layout()
plt.savefig("fig2_boxplot.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(original[["pollutant_min", "pollutant_max", "pollutant_avg",
                      "latitude", "longitude"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("fig3_heatmap.png", dpi=300)
plt.show()

# 7. Split data into training and testing sets
# one station has several rows, so we keep each station fully in train OR test
splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=df["station"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
print("\nTrain rows:", len(X_train), " Test rows:", len(X_test))

# 8. Create five regression models
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression": make_pipeline(
        StandardScaler(), PolynomialFeatures(degree=2), LinearRegression()),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
}

# 9. Train each model and test it
results = []
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })

# 10. Show the comparison table
results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\nModel Comparison:")
print(results_df.round(3).to_string(index=False))

# 11. Compare RMSE values
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["RMSE"])
plt.title("RMSE Comparison of Five Models (Lower is Better)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("fig4_rmse.png", dpi=300)
plt.show()

# 12. Compare R2 values
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["R2"], color="green")
plt.title("R2 Score Comparison of Five Models (Higher is Better)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("fig5_r2.png", dpi=300)
plt.show()

# 13. Actual vs predicted graph for the best model
best_model_name = results_df.iloc[0]["Model"]
best_predictions = predictions[best_model_name]
plt.figure(figsize=(6, 5))
plt.scatter(y_test, best_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Pollutant Average")
plt.ylabel("Predicted Pollutant Average")
plt.title("Actual vs Predicted - " + best_model_name)
plt.tight_layout()
plt.savefig("fig6_actual_vs_predicted.png", dpi=300)
plt.show()

# 14. Residual plot for the best model
residuals = y_test - best_predictions
plt.figure(figsize=(6, 5))
plt.scatter(best_predictions, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - " + best_model_name)
plt.tight_layout()
plt.savefig("fig7_residuals.png", dpi=300)
plt.show()

# 15. Save results
results_df.to_csv("model_comparison.csv", index=False)
print("\nBest model:", best_model_name)
print("Files saved: model_comparison.csv and 7 figure PNG files")
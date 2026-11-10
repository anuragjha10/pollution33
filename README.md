# Regression-Based Estimation of Average Air Pollutant Concentrations Across Indian Monitoring Stations

## Project Overview

This project focuses on estimating average air pollutant concentrations across monitoring stations in India using Machine Learning regression techniques.

The project uses geographical information (latitude and longitude) and pollutant types to predict average pollutant levels. Five different regression algorithms are implemented and compared to identify the best-performing model.

The project also includes data cleaning, exploratory data analysis (EDA), data visualization, model evaluation, and performance comparison.

## Objectives

- Analyze air quality data collected from monitoring stations across India.
- Clean and preprocess the dataset.
- Explore pollutant distributions and relationships using visualizations.
- Apply different regression algorithms to estimate average pollutant concentrations.
- Compare model performance using evaluation metrics.
- Identify the best-performing regression model.

## Technologies Used

- **Programming Language:** Python
- **Libraries:**
  - NumPy – Numerical computations
  - Pandas – Data manipulation and analysis
  - Matplotlib – Data visualization
  - Seaborn – Statistical visualization
  - Scikit-learn – Machine Learning and model evaluation

## Machine Learning Algorithms

The following five regression models are implemented:

1. **Linear Regression** – Models the linear relationship between input features and pollutant levels.
2. **Polynomial Regression** – Captures nonlinear relationships using polynomial features.
3. **Decision Tree Regression** – Predicts pollutant levels using decision-based rules.
4. **Random Forest Regression** – Combines multiple decision trees to improve prediction performance.
5. **K-Nearest Neighbors (KNN) Regression** – Predicts values based on the nearest data points.

## Dataset Description

The project uses an air quality dataset containing information about monitoring stations and pollutant measurements in India.

**Important columns:**

| Column | Description |
|---|---|
| station | Monitoring station name |
| latitude | Latitude of the monitoring station |
| longitude | Longitude of the monitoring station |
| pollutant_id | Type of pollutant (CO, NO2, PM2.5, etc.) |
| pollutant_min | Minimum pollutant level |
| pollutant_max | Maximum pollutant level |
| pollutant_avg | Average pollutant level (Target Variable) |

**Input Features:**
- Latitude
- Longitude
- Pollutant type (converted into numerical dummy variables)

**Target Variable:**
- pollutant_avg

## Project Workflow

1. **Data Loading:** Load the air quality dataset using Pandas.
2. **Data Understanding:** Examine dataset dimensions, data types, missing values, duplicates, and pollutant distribution.
3. **Data Cleaning:** Remove unnecessary columns, missing pollutant readings, and duplicate records.
4. **Feature Encoding:** Convert pollutant categories into numerical dummy variables using One-Hot Encoding.
5. **Feature Selection:** Select geographical coordinates and pollutant types as input features.
6. **Data Visualization:** Generate histograms, boxplots, and correlation heatmaps.
7. **Data Splitting:** Divide the dataset into 80% training data and 20% testing data using GroupShuffleSplit to keep stations separate between training and testing.
8. **Model Training:** Train five different regression models.
9. **Model Evaluation:** Evaluate models using MAE, MSE, RMSE, and R² Score.
10. **Model Comparison:** Compare the performance of all five models.
11. **Prediction Analysis:** Generate Actual vs Predicted and Residual plots for the best-performing model.
12. **Result Export:** Save model comparison results and visualization figures.

## Model Evaluation Metrics

| Metric | Description | Better Result |
|---|---|---|
| MAE | Mean Absolute Error | Lower |
| MSE | Mean Squared Error | Lower |
| RMSE | Root Mean Squared Error | Lower |
| R² Score | Measures how well the model explains variations in the target | Higher |

## Visualizations

The project generates seven visualizations:

1. Distribution of Average Pollutant Levels
2. Boxplot of Average Levels for Each Pollutant
3. Correlation Heatmap
4. RMSE Comparison of Five Regression Models
5. R² Score Comparison of Five Regression Models
6. Actual vs Predicted Values for the Best Model
7. Residual Plot for the Best Model

## Project Structure

```text
Air-Quality-Prediction/
│
├── air_quality.csv
├── main.py
├── model_comparison.csv
│
├── fig1_distribution.png
├── fig2_boxplot.png
├── fig3_heatmap.png
├── fig4_rmse.png
├── fig5_r2.png
├── fig6_actual_vs_predicted.png
├── fig7_residuals.png
│
└── README.md
```

## Installation and Setup

**1. Clone the repository**

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

**2. Navigate to the project directory**

```bash
cd Air-Quality-Prediction
```

**3. Install the required libraries**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**4. Run the Python program**

```bash
python main.py
```

Make sure `air_quality.csv` is available in the project directory before running the program.

## Output

After successful execution, the program generates:

- Model performance comparison table.
- Best-performing regression model.
- CSV file containing evaluation metrics.
- Seven graphical visualizations saved as PNG files.

## Future Scope

- Include additional environmental factors such as temperature, humidity, and wind speed.
- Explore advanced machine learning algorithms.
- Use time-series data for forecasting future air pollution levels.
- Improve prediction accuracy through hyperparameter tuning.
- Develop an interactive dashboard for visualizing air quality predictions.

## Author

**Anurag Kumar Jha**

B.Tech – Computer Science and Engineering  
Lovely Professional University

## Acknowledgements

This project was developed as part of academic work in Predictive Analytics, focusing on data analysis, regression techniques, and machine learning model evaluation.

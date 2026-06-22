# Pharmaceutical Sales Forecasting (PharmaCast)

Forecasting pharmaceutical sales is crucial for inventory planning, production scheduling, and marketing strategy. This project combines **GM(1,1) (Grey Model)** and **SVR (Support Vector Regression)** in a hybrid ensemble to predict future sales based on historical e-commerce data.

## 📌 Overview

PharmaCast uses a hybrid ensemble approach — combining the Grey Model GM(1,1) with Support Vector Regression (SVR) — to forecast category-wise pharmaceutical sales. The model accounts for trend and seasonal behavior using STL decomposition and combines both models using inverse-error weighting for improved accuracy.

## 🚀 Key Features

- **Hybrid Ensemble Model**: GM(1,1) + SVR combined using inverse-error weighting
- **Seasonality Handling**: STL (Seasonal-Trend decomposition using LOESS) applied to capture seasonal patterns
- **Category-wise Forecasting**: Trained and evaluated on multiple product categories
- **High Accuracy**: Achieved ~89.7% accuracy (Monsoon category) and ~91.8% accuracy (Vitamin C category)
- **Dataset**: 1,000-record pharmaceutical e-commerce sales dataset

## 🛠️ Tech Stack

- **Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn (SVR), Statsmodels (STL)
- **Visualization**: Power BI / Matplotlib
- **Deployment**: Streamlit (via Google Colab + ngrok)

## 📂 Project Structure

```
Pharmaceutical-Sales-Forecasting/
│
├── data/                   # Raw and processed dataset
├── notebooks/              # Jupyter notebooks for EDA & modeling
├── src/                    # Source code (GM(1,1), SVR, ensemble logic)
├── dashboard/              # Power BI / Streamlit dashboard files
├── reports/                # Project report & presentation
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Methodology

1. **Data Preprocessing**: Cleaning and structuring the e-commerce pharmaceutical sales dataset
2. **Seasonality Decomposition**: STL decomposition to separate trend, seasonal, and residual components
3. **GM(1,1) Modeling**: Grey Model applied for short-term trend forecasting
4. **SVR Modeling**: Support Vector Regression applied to capture non-linear patterns
5. **Ensemble Weighting**: Final forecast computed using inverse-error weighting of GM(1,1) and SVR outputs
6. **Evaluation**: Model accuracy validated category-wise (e.g., Monsoon, Vitamin C)

## 📊 Results

| Category | Accuracy |
|----------|----------|
| Monsoon | ~89.7% |
| Vitamin C | ~91.8% |

## 📈 Dashboard

An interactive dashboard (Power BI-style, Python(MatplotLib,Seaborn) is included to visualize sales trends, forecasts, and category-wise performance.

## 🔧 Installation

```bash
git clone https://github.com/TauhidShaikh517/Pharmaceutical-Sales-Forecasting.git
cd Pharmaceutical-Sales-Forecasting
pip install -r requirements.txt
```

## ▶️ Usage

```bash
python src/main.py
```

## 📄 Project Report

Detailed methodology, literature review, and results are available in the `reports/` folder.

## 👤 Author

**Tauhid Shaikh**
- GitHub: [@TauhidShaikh517](https://github.com/TauhidShaikh517)
- LinkedIn: [tauhid-shaikh](https://www.linkedin.com/in/tauhid-shaikh-88ba52266/)

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

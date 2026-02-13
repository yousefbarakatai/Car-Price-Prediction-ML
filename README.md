 # 💎🚘 Car Selling Price Prediction & Market Analysis Dashboard

## 🎯 Objectives

* Identify key factors affecting car prices.
* Build and optimize regression models.
* Deploy an interactive dashboard & prediction application.
* Apply production-ready ML best practices.

## 📂 Dataset

The dataset includes detailed information about used cars:

* **Specs**: Model Year, KM Driven, Mileage, Engine Capacity, Max Power, Seats.
* **Features**: Fuel Type, Seller Type, Transmission, Owner.
* **Target**: Selling Price.

## 🛠 Data Preprocessing

### ✔ Data Cleaning

* Removed measurement units (kmpl, CC, bhp).
* Extracted numeric values from text-based columns.
* Handled missing values using median imputation.
* Removed duplicate records.
* Filtered invalid values (e.g., zero mileage).

### ✔ Feature Engineering

* Created **car_age** feature.
* Applied **Ordinal Encoding** for owner category.
* Applied **One-Hot Encoding** for categorical variables (Fuel, Seller, Transmission).
* Standardized numerical features using **StandardScaler**.
* Built a clean **Scikit-Learn Pipeline**.

## 📊 Exploratory Data Analysis (EDA)

The analysis explored:

* Price distribution & Price vs KM Driven (log-scale visualization).
* Yearly price trends & Brand-based price comparison.
* Correlation heatmap to identify key price drivers.

## 🔎 Key Insights

* **Car age** and **max power** strongly influence price.
* **Diesel** vehicles often retain higher resale value.
* **Automatic** transmission vehicles tend to be more expensive.

## 🤖 Machine Learning

### Models Tested:

* Linear Regression
* Lasso Regression
* Decision Tree Regressor
* **Random Forest Regressor (Champion Model)**

### Hyperparameter Tuning:

* Performed using **GridSearchCV** (CV = 3/5).
* Model selection based on **R² Score** (Achieved ~88%).

## 🏗 Production Pipeline

The final model uses a Scikit-Learn Pipeline:
`StandardScaler` ➔ `Optimized Random Forest`
This ensures clean preprocessing, no data leakage, and simplified deployment.

## 📊 Interactive Dashboard

Built using **Streamlit + Plotly**.

* **Market KPIs**: Avg Price, Total Cars, Avg Age.
* **Dynamic Filters**: Slice data by fuel, seller, transmission, and owner.
* **Visual Charts**: Mileage impact, brand price comparison, and yearly trends.

## 💰 Price Prediction Module

Users can enter car specifications and get a **real-time price estimation** with an interactive and formatted UI.

## 🚀 Run Locally

1. Clone the repo: `git clone <your-repo-link>`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📁 Project Structure

```text
├── cleaned_cars_dashboard.csv  # Processed dataset
├── model.pkl                  # Serialized optimized model
├── scaler.pkl                 # Serialized scaler object
├── app.py                     # Streamlit application
├── notebook.ipynb             # Full training code
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation

```

## 👨‍💻 Author

**Yousef Barakat**
*Data Analyst & Machine Learning Enthusiast*
📧 ya139471@gmail.com
🔗 [LinkedIn Profile](https://www.linkedin.com/in/yousef-ahmed-95868b276/)

---

*Generated with ❤️ for the Data Science Community.*

---
 
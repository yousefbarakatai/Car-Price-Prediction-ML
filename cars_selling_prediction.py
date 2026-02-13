# # 💎🚘 CARS SELLING PREDICTION


# ![Car Image](image_car.jpg)
# 


# # 🗺️ Project Execution Roadmap
# 
# * **[📥 Environment & Data Loading](#section1)**
#     * [📦 Importing Libraries](#section1_1)
#     * [📂 CSV Data Ingestion](#section1_2)
#     * [🕵️ Initial Data Inspection](#section1_3)
# 
# * **[🧹 Data Cleaning](#section2)**
#     * [✂️ Unit Stripping (Regex)](#section2_1)
#     * [🩹 Missing Value Imputation](#section2_2)
#     * [🚫 Duplicate Removal](#section2_3)
# 
# * **[🔍 Exploratory Data Analysis (EDA)](#section3)**
#     * [🔥 Correlation Heatmaps](#section3_1)
#     * [📌 Top Brands Analysis](#section3_2)
# 
# * **[⚙️ Feature Engineering](#section4)**
#     * [📉 Outlier Filtering](#section4_1)
#     * [🔢 Ordinal & One-Hot Encoding](#section4_2)
#     * [🎯 Target & Features Splitting](#section4_3)
#     * [⚖️ Feature Scaling](#section4_4)
# 
# * **[🤖 Machine Learning Modeling](#section5)**
#     * [🧪 Multiple Model Testing](#section5_1)
#     * [🎯 Hyperparameter Tuning (GridSearch)](#section5_1)
#     * [🏆 Best Model Selection (Random Forest)](#section5_2)
#     * [💾 Model Export](#section5_3)
# 
# * **[🤖 Data Dashboard](#section6)**
# 
# 
# ---
# <a id='main_menu'></a>


# <a id='section1'></a>
# # 📂 Loading & Inspecting Data


# <a id='section1_1'></a>
# *📦 Importing Libraries


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib


from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.pipeline import Pipeline


# <a id='section1_2'></a>
# * 📂 CSV Data Ingestion


df = pd.read_csv(r'C:\Users\Enter Computer\Downloads\Project_04_reg_cars_selling/reg_cars_selling.csv')


# [↑ Back to Roadmap](#main_menu)


# <a id='section1_3'></a>
# * 🕵️ Initial Data Inspection


df.head()


df.info()


pd.set_option('display.float_format', '{:.2f}'.format)



df.describe()


# [↑ Back to Roadmap](#main_menu)


# <a id='section2'></a>
# # 🛠️ Data Cleaning
# 


# <a id='section2_1'></a>
# * 🪓 String Manipulation


df['mileage'].str.contains('km/kg').sum()


df['mileage'] = df['mileage'].str.replace('kmpl',' ').str.replace('km/kg',' ').astype(float)


df['engine'] = df['engine'].str.replace('CC',' ').astype(float)


df['max_power'] = df['max_power'].str.replace('bhp',' ')


df['max_power'] = (
    df['max_power']
    .astype(str)
    .str.extract(r'([\d.]+)')[0]
    .astype(float)
)



df.drop(columns='torque',inplace=True)


df['brand'] = df['name'].str.split(' ').str[0]


df['seats'] = df['seats'].astype('Int32')


# [↑ Back to Roadmap](#main_menu)


# <a id='section2_2'></a>
# * ❓ Missing Values 
# 


df.isna().sum()


df['mileage'].fillna(df['mileage'].median(),inplace=True)


df['engine'].fillna(df['engine'].median(),inplace=True)


df['max_power'].fillna(df['max_power'].median(),inplace=True)


df['seats'].fillna(df['seats'].median(),inplace=True)


df.isna().sum()


# [↑ Back to Roadmap](#main_menu)


# <a id='section2_3'></a>
# * 🔄 Duplicates 


df.duplicated().sum()


df.drop_duplicates(inplace=True, keep='first')


df.duplicated().sum()


df = df[df['mileage'] > 0]
df = df[df['max_power'] > 0]
df = df[df['seats'] < 14]


# [↑ Back to Roadmap](#main_menu)


# <a id='section3'></a>
# # 🔍 EDA
# 


# Selling price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['selling_price'], kde=True, color='blue')
plt.title('Distribution of Selling Price', fontsize=14, fontweight='bold')
plt.show()


# The Most Expensive Car
df[df['selling_price'] == df['selling_price'].max()]


# The Cheapest Car
df[df['selling_price'] == df['selling_price'].min()]


# The Most Used Car (Max KM)
df[df['km_driven'] == df['km_driven'].max()]


# The lowst Used Car (Max KM)
df[df['km_driven'] == df['km_driven'].min()]


# KM Driven vs Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='km_driven', y='selling_price', alpha=0.4, color='purple')
plt.xscale('log') # لتقريب النقاط من بعضها
plt.yscale('log')
plt.title('KM Driven vs Selling Price', fontsize=14, fontweight='bold')
plt.show()


# Count of Cars by Year
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='year', palette='viridis')
plt.xticks(rotation=45)
plt.title('Count of Cars by Year', fontsize=14, fontweight='bold')
plt.show()


# Newest Model Year
df[df['year'] == df['year'].max()].head()


# Oldest Model Year
df[df['year'] == df['year'].min()]


# Average Selling Price Trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='year', y='selling_price', estimator='mean', marker='o')
plt.title('Average Selling Price Trend', fontsize=14, fontweight='bold')
plt.show()


# Fuel Type
df['fuel'].value_counts()


# Distribution of Fuel Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='fuel', palette='Set2')
plt.title('Distribution of Fuel Types', fontsize=14, fontweight='bold')
plt.show()


# Selling Price vs Fuel
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='fuel', y='selling_price', palette='Set2')
plt.yscale('log') 
plt.title('Selling Price vs Fuel Type', fontsize=14, fontweight='bold')
plt.show()


# Price Trend per Year (by Fuel Type)
plt.figure(figsize=(12, 7))
sns.lineplot(data=df, x='year', y='selling_price', hue='fuel', marker='o', errorbar=None)
plt.title('Price Trend per Year (by Fuel Type)', fontsize=14, fontweight='bold')
plt.ylabel('Average Selling Price')
plt.show()


# Seller Type
df['seller_type'].value_counts()


# verage Selling Price by Seller Type
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='seller_type', y='selling_price', palette='muted')
plt.title('Average Selling Price by Seller Type')
plt.show()


# Transmission Type Count
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='transmission', palette='Set1')
plt.title('Transmission Type Count', fontsize=14, fontweight='bold')
plt.show()


# Selling Price vs Transmission
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='transmission', y='selling_price', palette='Set1')
plt.yscale('log')
plt.title('Selling Price vs Transmission', fontsize=14, fontweight='bold')
plt.show()


# Owner Types 
df['owner'].value_counts()


# Distribution of Owner Types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='owner', palette='pastel', order=df['owner'].value_counts().index)
plt.title('Distribution of Owner Types', fontsize=14, fontweight='bold')
plt.show()


# Selling Price vs owner
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='owner', y='selling_price', palette='Set1')
plt.yscale('log')
plt.title('Selling Price vs owner', fontsize=14, fontweight='bold')
plt.show()


# Seats
df['seats'].value_counts()


# Influence of Number of Seats on Price
plt.figure(figsize=(10, 6))
sns.pointplot(data=df, x='seats', y='selling_price', color='red')
plt.title('Influence of Number of Seats on Price')
plt.show()


# Brand type
df['brand'].value_counts()


from matplotlib.ticker import ScalarFormatter
# Price Range Distribution for Top 10 Most Common Brands
top_10_brands_list = df['brand'].value_counts().head(10).index
df_top_10 = df[df['brand'].isin(top_10_brands_list)]

plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

ax = sns.boxplot(data=df_top_10, x='brand', y='selling_price', palette='Spectral', order=top_10_brands_list)

plt.yscale('log')  
plt.title('Price Range Distribution for Top 10 Most Common Brands', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Car Brand', fontsize=12)
plt.ylabel('Selling Price (Log Scale)', fontsize=12)

ax.yaxis.set_major_formatter(ScalarFormatter())
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Average Price per Brand (by Transmission)
top_5_brands = df['brand'].value_counts().head(10).index
df_top_5 = df[df['brand'].isin(top_5_brands)]

plt.figure(figsize=(12, 7))
sns.barplot(data=df_top_5, x='brand', y='selling_price', hue='transmission', palette='Set2')
plt.title('Average Price per Brand (by Transmission)', fontsize=14, fontweight='bold')
plt.show()


# <a id='section3_2'></a>
# * Top 10 Car Brands


# Top 10 Car Brands
top_10_counts = df['brand'].value_counts().head(10)
 
top_10_price = df.groupby('brand')['selling_price'].mean().sort_values(ascending=False).head(10)
 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=top_10_counts.values, y=top_10_counts.index, palette='viridis')
plt.title('Top 10 Car Brands by Count')
plt.xlabel('Number of Cars')

plt.subplot(1, 2, 2)
sns.barplot(x=top_10_price.values, y=top_10_price.index, palette='magma')
plt.title('Top 10 Most Expensive Brands (Avg Price)')
plt.xlabel('Average Price')

plt.tight_layout()
plt.show()


df[df['brand'].isin(['Volvo', 'Lexus'])]


# Max Power vs Selling Price
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='max_power', y='selling_price', alpha=0.4, color='green')
plt.title('Max Power vs Selling Price', fontsize=14, fontweight='bold')
plt.show()


# <a id='section3_1'></a>
# * Correlation Heatmap


# Correlation Heatmap
corr = df[[ 'selling_price', 'year', 'km_driven' , 'mileage', 'engine', 'max_power', 'seats']].corr()

 
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap: What affects the Selling Price?')
plt.show()


# [↑ Back to Roadmap](#main_menu)


# # 🛠️ Feature Engineering
# <a id='section4'></a>


# * ✖️ Outliers
# <a id='section4_1'></a>


df.describe()


columns = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power']
df[columns].plot(kind='box', subplots=True, layout=(2, 3), figsize=(15, 10), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


#Q1 = df['selling_price'].quantile(0.25)
#Q3 = df['selling_price'].quantile(0.75)
#IQR = Q3 - Q1

#lower = Q1 - 1.5 * IQR
#upper = Q3 + 1.5 * IQR

#df = df[(df['selling_price'] >= lower) & (df['selling_price'] <= upper)]


#Q1 = df['km_driven'].quantile(0.25)
#Q3 = df['km_driven'].quantile(0.75)
#IQR = Q3 - Q1

#lower = Q1 - 1.5 * IQR
#upper = Q3 + 1.5 * IQR

#df = df[(df['km_driven'] >= lower) & (df['km_driven'] <= upper)]


#Q1 = df['engine'].quantile(0.25)
#Q3 = df['engine'].quantile(0.75)
#IQR = Q3 - Q1

#lower = Q1 - 1.5 * IQR
#upper = Q3 + 1.5 * IQR

#df = df[(df['engine'] >= lower) & (df['engine'] <= upper)]


#Q1 = df['mileage'].quantile(0.25)
#Q3 = df['mileage'].quantile(0.75)
#IQR = Q3 - Q1

#lower = Q1 - 1.5 * IQR
#upper = Q3 + 1.5 * IQR

#df = df[(df['mileage'] >= lower) & (df['mileage'] <= upper)]


#Q1 = df['max_power'].quantile(0.25)
#Q3 = df['max_power'].quantile(0.75)
#IQR = Q3 - Q1

#lower = Q1 - 1.5 * IQR
#upper = Q3 + 1.5 * IQR

#df = df[(df['max_power'] >= lower) & (df['max_power'] <= upper)]


# [↑ Back to Roadmap](#main_menu)


# * 🏷️ Encoding 
# <a id='section4_2'></a>
# 


owner_encoder = OrdinalEncoder(
    categories=[[
        'First Owner',
        'Second Owner',
        'Third Owner',
        'Fourth & Above Owner',
        'Test Drive Car'
    ]]
)

df[['owner_enc']] = owner_encoder.fit_transform(df[['owner']])



enc = OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)

ohe = enc.fit_transform(df[['fuel', 'seller_type', 'transmission']])

ohe_cols = enc.get_feature_names_out(['fuel', 'seller_type', 'transmission'])

df[ohe_cols] = ohe

 



df


# [↑ Back to Roadmap](#main_menu)


# * 📂 Splitting Data
# <a id='section4_3'></a>


current_year = datetime.now().year
df['car_age'] = current_year - df['year']


X = df.drop(['name','year','selling_price','fuel','seller_type','transmission','owner','brand'],axis=1)
y = df['selling_price']


X


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# * ⚖️ Scaling
# <a id='section4_4'></a>


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # بنعمل transform بس للـ test
 
joblib.dump(scaler, 'scaler.pkl')


# [↑ Back to Roadmap](#main_menu)


# # 💡 Building the Model
# <a id='section5'></a>


# <a id='section5_1'></a>
# * 🎯 Hyperparameter Tuning


# 1️⃣ Linear Regression
param_lr = {
    'regressor': [LinearRegression()]
    # غالباً ملوش hyperparameters للتعديل
}

# 2️⃣ Lasso
param_lasso = {
    'regressor': [Lasso(max_iter=5000, random_state=42)],
    'regressor__alpha': [0.1, 0.01, 0.001]
}

# 3️⃣ Decision Tree Regressor
param_dt = {
    'regressor': [DecisionTreeRegressor(random_state=42)],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# 4️⃣ Random Forest
param_rf = {
    'regressor': [RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )],
    'regressor__n_estimators': [100, 300, 500],
    'regressor__max_depth': [None, 15, 25],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 'log2']
}


# * 🏆 Best Model Selection
# <a id='section5_2'></a>


# Pipeline Setup
pipe = Pipeline([
    ('regressor', LinearRegression()) # البداية بموديل افتراضي
])

params = [param_lr, param_lasso, param_dt, param_rf]

# GridSearchCV
gs = GridSearchCV(
    pipe,
    params,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=2
)

gs.fit(X_train_scaled, y_train)


gs.best_params_


gs.best_score_


# * 💾 Model Export
# <a id='section5_3'></a>


joblib.dump(gs.best_estimator_,'model.pkl')


my_model = joblib.load('model.pkl')

X_train.head(5)


input_data = np.array([[160000, 20,1000, 100, 5, 1, 0,0,0,1,1,0,0,0,1,5]])

input_data_scaled = scaler.transform(input_data)
input_data_scaled


prediction_scaled = my_model.predict(input_data_scaled)
prediction_scaled


# [↑ Back to Roadmap](#main_menu)


# <a id='section6'></a>
# # Dashboard


df_dash = df[['name','year','km_driven',
             'fuel','seller_type',
             'transmission','owner',
             'mileage','engine',
             'max_power','seats','brand',
             'car_age','selling_price']]


df_dash.to_csv('cleaned_cars_dashboard.csv', index=False)



df_dash


# 












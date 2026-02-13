import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
 

st.set_page_config(
    page_title="Car Price Prediction",
    layout="centered",
    page_icon="📊"
)

if "page" not in st.session_state:
    st.session_state.page = "prediction"

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Dashboard"):
        st.session_state.page = "dashboard"

with col2:
    if st.button("🚗 Prediction"):
        st.session_state.page = "prediction"

# Dashboard Page
# ----------------------------------
if st.session_state.page == "dashboard":

    # 📊 Dashboard
    # load data

    @st.cache_data
    def load_data():
        df = pd.read_csv("cleaned_cars_dashboard.csv")
        return df

    df = load_data()

    # sidebar 

    with st.sidebar:
        st.header("Cars Dashboard")
        st.image("image_car.jpg", width=200)
        st.write("An interactive dashboard to explore used car prices and understand the key factors influencing car value.")

        st.subheader("Filters")
        fuel_filter = st.multiselect(
            "Fuel",
            options=df["fuel"].unique(),
            default=df["fuel"].unique()
        )

        seller_filter = st.multiselect(
            "Seller Type",
            options=df["seller_type"].unique(),
            default=df["seller_type"].unique()
        )

        transmission_filter = st.multiselect(
            "Transmission",
            options=df["transmission"].unique(),
            default=df["transmission"].unique()
        )

        owner_filter = st.multiselect(
            "Owner",
            options=df["owner"].unique(),
            default=df["owner"].unique()
        )
    
        st.subheader("Contact")
        st.markdown("Made with: by Eng. [Yousef Barakat](https://www.linkedin.com/in/yousef-ahmed-95868b276/)")
        st.write("📧 ya139471@gmail.com")
        st.write("📞 01032037435")

    # Apply filters
    if not fuel_filter:
        fuel_filter = df["fuel"].unique()

    if not seller_filter:
        seller_filter = df["seller_type"].unique()

    if not transmission_filter:
        transmission_filter = df["transmission"].unique()

    if not owner_filter:
        owner_filter = df["owner"].unique()


    df_dash = df[
        (df["fuel"].isin(fuel_filter)) &
        (df["seller_type"].isin(seller_filter)) &
        (df["transmission"].isin(transmission_filter)) &
        (df["owner"].isin(owner_filter))
    ]

    # ---------------- Page Config ----------------

    st.header("📊 Car Market Dashboard")

    # ---------------- KPIs ----------------
    avg_price = df_dash["selling_price"].mean()
    total_cars = df_dash.shape[0]
    avg_age = df_dash["car_age"].mean()
    avg_km = df_dash["km_driven"].mean()

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Average Price", f"{avg_price:,.0f}")
    k2.metric("Total Cars", total_cars)
    k3.metric("Average Car Age (Year)", f"{avg_age:.1f}")
    k4.metric("Average KM Driven", f"{avg_km:,.0f}")


    # ---------------- Visuals ----------------
    c1, c2 = st.columns(2)

    # fig1
    price_year = (
        df_dash
        .groupby("year")["selling_price"]
        .mean()
        .reset_index()
    )

    fig_year = px.line(
        price_year,
        x="year",
        y="selling_price",
        title="Average Selling Price Over Years"
    )
    c1.plotly_chart(fig_year, use_container_width=True)

    # fig2
    top_brands = (
        df_dash["brand"]
        .value_counts()
        .head(10)
        .index
    )

    brand_price = (
        df_dash[df_dash["brand"].isin(top_brands)]
        .groupby("brand")["selling_price"]
        .mean()
        .reset_index()
    )

    fig_brand = px.bar(
        brand_price,
        x="brand",
        y="selling_price",
        title="Average Price by Brand (Top 10 Brands)"
   )

    c2.plotly_chart(fig_brand, use_container_width=True)

    c3, c4 = st.columns(2)

    #fig3
    df_dash['mileage_range'] = pd.cut(df_dash['mileage'], bins=5)
    df_dash['mileage_range'] = pd.cut(df_dash['mileage'], bins=5).astype(str)

    mileage_price = (
        df_dash
        .groupby("mileage_range")["selling_price"]
        .mean()
        .reset_index()
    )

    fig_mileage = px.bar(
        mileage_price,
        x='mileage_range',
        y="selling_price",
        title="Average Price by Mileage Range"
    )

    c3.plotly_chart(fig_mileage, use_container_width=True)

    #fig4
    seats_price = (
        df_dash
        .groupby("seats")["selling_price"]
        .mean()
        .reset_index()
    )

    fig_seats = px.bar(
        seats_price,
        x="seats",
        y="selling_price",
        title="Average Price by Number of Seats"
    )

    c4.plotly_chart(fig_seats, use_container_width=True)

# ----------------------------------
# Prediction Page (Placeholder)
# ----------------------------------
elif st.session_state.page == "prediction":

    with st.sidebar:
        st.header("Car Price Prediction")

        st.image("image_car.jpg", width=200)

        st.write("""
        Enter the car specifications to get an estimated market price 
        based on our trained Machine Learning model.
        
        The prediction is powered by an optimized Random Forest model
        trained on real used car market data.
        """)

        st.subheader("🧠 Model Info")
        st.write("• Model: Random Forest Regressor")
        st.write("• Tuned with GridSearchCV")
        st.write("• Evaluation Metric: R² Score")
        st.write("• Preprocessing: StandardScaler + Pipeline")
 
        st.subheader("👨‍💻 About the Developer")
        st.markdown("Developed by Eng. [Yousef Barakat](https://www.linkedin.com/in/yousef-ahmed-95868b276/)")
        st.write("📧 ya139471@gmail.com")
        st.write("📞 01032037435")

        st.caption("This prediction is for estimation purposes only and may vary from actual market prices.")


    # ---------------- Page Config ----------------

    st.title("CAR PRICE PREDICTION")
    st.image("image_car.jpg")
    st.write("Enter the car details to predict its selling price")

    # ---------------- Inputs ----------------
    year = st.slider("Model Year", 1950, 2050, 2014)
    km_driven = st.number_input("KM Driven", min_value=0)
    mileage = st.number_input("Mileage (kmpl)", min_value=0.0)
    engine = st.number_input("Engine (CC)", min_value=500)
    max_power = st.number_input("Max Power (bhp)", min_value=0.0)
    seats = st.selectbox("Seats", [2, 4, 5,6, 7, 8,9,10])

    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller_type = st.selectbox(
        "Seller Type",
        ["Individual", "Dealer", "Trustmark Dealer"]
    )
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox(
        "Owner",
        [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car"
        ]
    )

    btn = st.button("💰 Predict Price")
    # ---------------- Prediction ----------------
    if btn:
        # load model and scaler
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        
        # encoding
        FEATURE_ORDER = [
            'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'owner_enc',
            'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
            'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
            'transmission_Automatic', 'transmission_Manual', 'car_age'
        ]

        #  
        current_year = 2024
        car_age_val = current_year - year

        owner_mapping = {
            "First Owner": 0, "Second Owner": 1, "Third Owner": 2,
            "Fourth & Above Owner": 3, "Test Drive Car": 4
        }
 
        input_dict = {col: 0 for col in FEATURE_ORDER}
        
        input_dict['km_driven'] = km_driven
        input_dict['mileage'] = mileage
        input_dict['engine'] = engine
        input_dict['max_power'] = max_power
        input_dict['seats'] = seats
        input_dict['owner_enc'] = owner_mapping[owner]
        input_dict['car_age'] = car_age_val

        input_dict[f"fuel_{fuel}"] = 1
        input_dict[f"seller_type_{seller_type}"] = 1
        input_dict[f"transmission_{transmission}"] = 1

        X_input = pd.DataFrame([input_dict])[FEATURE_ORDER]

        # predict

        X_scaled = scaler.transform(X_input)

        prediction = model.predict(X_scaled)[0]

        st.markdown("---")
 
        final_price = max(0, prediction)
        st.success(f"### 💰 Estimated Car Price: {final_price:,.0f} EGP")
        st.balloons()
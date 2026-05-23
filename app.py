import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)



try:
    df = pd.read_csv("House_price_prediction.csv")
except:
    st.error("Dataset not found! Please place 'House_price_prediction.csv' in the same folder.")
    st.stop()



X = df[['bedrooms', 'bathrooms', 'sqft_living']]
y = df['price']

model = LinearRegression()
model.fit(X, y)



st.markdown("""
# 🏠 House Price Prediction App
### Simple Machine Learning Demo
---
""")



st.sidebar.header("Enter House Details")

bedrooms = st.sidebar.number_input("Bedrooms", 1, 10, 3)
bathrooms = st.sidebar.number_input("Bathrooms", 1, 5, 2)
sqft = st.sidebar.number_input("Square Feet", 300, 5000, 1500)

predict_btn = st.sidebar.button("Predict Price 🔮")



predicted_price = None

if predict_btn:
    input_data = np.array([[bedrooms, bathrooms, sqft]])
    predicted_price = model.predict(input_data)[0]


st.subheader("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Houses", len(df))
col2.metric("Average Price", f"₹ {int(df['price'].mean()):,}")
col3.metric("Max Price", f"₹ {int(df['price'].max()):,}")



if predicted_price is not None:
    st.success("Prediction Complete!")

    st.markdown(
        f"""
        <div style="
            background-color:#d1ecf1;
            padding:25px;
            border-radius:10px;
            text-align:center;
            font-size:26px;
            font-weight:bold;">
            💰 Estimated Price: ₹ {int(predicted_price):,}
        </div>
        """,
        unsafe_allow_html=True
    )



st.markdown("---")
st.subheader("📈 Price vs Area")

fig, ax = plt.subplots()

ax.scatter(df["sqft_living"], df["price"], alpha=0.3)

if predicted_price:
    ax.scatter(sqft, predicted_price, color="red", s=200)

ax.set_xlabel("Square Feet")
ax.set_ylabel("Price")

st.pyplot(fig)



st.subheader("🏘 Bedrooms vs Price")

fig, ax = plt.subplots()

ax.scatter(df["bedrooms"], df["price"], alpha=0.3)

ax.set_xlabel("Bedrooms")
ax.set_ylabel("Price")

st.pyplot(fig)


with st.expander("View Dataset"):
    st.dataframe(df.head(15))

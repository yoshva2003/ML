
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache
def load_data():
    file_path = 'Salary_Data.csv'
    data = pd.read_csv(file_path)
    return data

data = load_data()

X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)
mean_train = np.mean(y_train)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_test_pred)
mean_test = np.mean(y_test)

st.title("Salary Prediction App")

experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)
if experience:
    salary_pred = model.predict([[experience]])
    st.write(f"Predicted Salary for {experience} years of experience: ₹{salary_pred[0]:,.2f}")

if st.button("Mean"):
    st.write(f"Mean Salary (Training): ₹{mean_train:,.2f}")
    st.write(f"Mean Salary (Testing): ₹{mean_test:,.2f}")

if st.button("MSE"):
    st.write(f"MSE (Training): {mse_train:,.2f}")
    st.write(f"MSE (Testing): {mse_test:,.2f}")

if st.button("RMSE"):
    st.write(f"RMSE (Training): {rmse_train:,.2f}")
    st.write(f"RMSE (Testing): {rmse_test:,.2f}")

if st.button("R²"):
    st.write(f"R² (Training): {r2_train:.2%}")
    st.write(f"R² (Testing): {r2_test:.2%}")

if st.button("Visualizations"):
    st.write("### Actual vs Predicted Salaries")
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(X_train, y_train, color='blue', label='Actual')
    axs[0].plot(X_train, y_train_pred, color='red', label='Predicted')
    axs[0].set_title("Training Data")
    axs[0].set_xlabel("Years of Experience")
    axs[0].set_ylabel("Salary")
    axs[0].legend()

    axs[1].scatter(X_test, y_test, color='green', label='Actual')
    axs[1].plot(X_test, y_test_pred, color='orange', label='Predicted')
    axs[1].set_title("Test Data")
    axs[1].set_xlabel("Years of Experience")
    axs[1].set_ylabel("Salary")
    axs[1].legend()

    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
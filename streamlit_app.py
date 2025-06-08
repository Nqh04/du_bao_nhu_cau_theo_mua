import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler # Assuming StandardScaler was used

# Load the trained model and scaler
# Make sure the paths match where you saved the files
try:
    gb_model_clf = joblib.load('gradient_boosting_model_clf.pkl')
    scaler_classification = joblib.load('scaler_classification.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please make sure 'gradient_boosting_model_clf.pkl' and 'scaler_classification.pkl' are in the same directory.")
    st.stop()

# Define possible values for categorical features based on the original dataset
# These should match the columns created during one-hot encoding during training
# You need to get these lists from your data exploration/preprocessing steps
season_options = ['Hè', 'Đông', 'Xuân', 'Thu']
gender_options = ['Male', 'Female']
occupation_options = ['Nghề tự do', 'Nhân viên văn phòng', 'Sinh viên, học sinh', 'Nhà đầu tư', 'Công nhân', 'Nhân viên part-time', 'Giám đốc', 'Trưởng phòng', 'Thực tập sinh', 'Doanh nhân', 'Bác sĩ', 'Kỹ sư', 'Luật sư', 'Kiến trúc sư', 'Nghệ sĩ', 'Giáo viên', 'Chủ cửa hàng', 'Quản lý']
store_location_options = ['Nha Trang - City Center', 'Da Nang - Thanh Khe', 'Hue - City Center', 'Da Nang - Hai Chau', 'Can Tho - Ninh Kieu', 'Ho Chi Minh City - District 7', 'Ho Chi Minh City - Tan Binh', 'Hanoi - Ba Dinh', 'Hanoi - Cau Giay', 'Ho Chi Minh City - District 1', 'Ho Chi Minh City - District 3', 'Vung Tau - City Center'] # Add all unique store locations

# Streamlit app title and description
st.title("Highland Coffee Product Predictor")
st.write("Enter the customer and transaction details to predict the likely product name.")

# Create input fields for user input
st.header("Enter Details:")

selected_season = st.selectbox("Season", season_options)
selected_gender = st.selectbox("Gender", gender_options)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
selected_occupation = st.selectbox("Occupation", occupation_options)
total_bill = st.number_input("Total Bill", min_value=0.0, value=50000.0)
selected_store_location = st.selectbox("Store Location", store_location_options)

# Prediction button
if st.button("Predict Product"):
    # Create a dictionary with user input
    user_input = {
        'Season': selected_season,
        'Gender': selected_gender,
        'Age': age,
        'Occupation': selected_occupation,
        'Total_Bill': total_bill,
        'store_location': selected_store_location
    }

    # Convert user input to a pandas DataFrame
    input_df = pd.DataFrame([user_input])

    # --- Preprocessing user input ---
    # Apply one-hot encoding to categorical features
    # Need to ensure all possible categories from training are considered to match columns
    for col in season_options:
        input_df[f'Season_{col}'] = (input_df['Season'] == col).astype(int)
    for col in gender_options:
         input_df[f'Gender_{col}'] = (input_df['Gender'] == col).astype(int)
    for col in occupation_options:
         input_df[f'Occupation_{col}'] = (input_df['Occupation'] == col).astype(int)
    for col in store_location_options:
         input_df[f'store_location_{col}'] = (input_df['store_location'] == col).astype(int)

    # Drop the original categorical columns
    input_df = input_df.drop(columns=['Season', 'Gender', 'Occupation', 'store_location'])

    # Rename columns to match the format from training (if drop_first=True was used, handle accordingly)
    # The column names should exactly match X_train_classification
    # A robust way is to get the column names from X_train_classification and reindex input_df
    # For this example, we'll assume drop_first=True wasn't used for simplicity in this part,
    # but in a real scenario, you'd need to adjust column names based on drop_first=True

    # Assuming X_train_classification column names are available (replace with actual names)
    # For demonstration, let's assume the one-hot encoded columns are named like 'Season_Hè', 'Gender_Male', etc.
    # and numerical columns are 'Age', 'Total_Bill'

    # Reorder columns to match the training data - this is crucial!
    # Get the list of columns from X_train_classification after preprocessing
    # Assuming X_train_classification is available in the environment (replace with loading if needed)
    # For now, manually construct expected column names based on the preprocessing steps
    expected_columns = ['Age', 'Total_Bill'] # numerical columns
    # Add one-hot encoded columns (handling drop_first=True if applicable)
    for col in season_options[1:]: # Assuming drop_first=True was used for Season
        expected_columns.append(f'Season_{col}')
    for col in gender_options[1:]: # Assuming drop_first=True was used for Gender
        expected_columns.append(f'Gender_{col}')
    for col in occupation_options[1:]: # Assuming drop_first=True was used for Occupation
        expected_columns.append(f'Occupation_{col}')
    for col in store_location_options[1:]: # Assuming drop_first=True was used for store_location
        expected_columns.append(f'store_location_{col}')

    # Ensure input_df has all expected columns, adding 0 if a category is not present
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reindex the input_df to match the order of columns in X_train_classification
    input_df = input_df[expected_columns]


    # Scale numerical features using the loaded scaler
    numerical_cols_classification = ['Age', 'Total_Bill']
    input_df[numerical_cols_classification] = scaler_classification.transform(input_df[numerical_cols_classification])

    # --- Make prediction ---
    prediction = gb_model_clf.predict(input_df)

    # Display the prediction
    st.subheader("Predicted Product Name:")
    st.success(prediction[0])

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import OneHotEncoder

# Load the trained model and scaler
try:
    gb_model_clf = joblib.load('gradient_boosting_model_clf.pkl')
    scaler_classification = joblib.load('scaler_classification.pkl')
    # Assuming the encoder was saved with a specific name, load it here
    # If you saved the encoder in a different way or with a different name, adjust this line
    encoder = joblib.load('onehot_encoder.joblib')
except FileNotFoundError:
    st.error("Model, scaler, or encoder file not found. Please make sure 'gradient_boosting_model_clf.pkl', 'scaler_classification.pkl', and 'onehot_encoder.joblib' are in the same directory.")
    st.stop()

# Define possible values for categorical features based on the original dataset
# These should match the categories the OneHotEncoder was fitted on
# It's best to get these from the fitted encoder if possible, but for a standalone app,
# you might need to hardcode them based on your training data.
# Let's load the encoder and get the categories from it
try:
    categorical_features_list = ['Season', 'Gender', 'Occupation', 'store_location'] # Original categorical column names
    # Get the categories from the loaded encoder
    season_options = encoder.categories_[categorical_features_list.index('Season')].tolist()
    gender_options = encoder.categories_[categorical_features_list.index('Gender')].tolist()
    occupation_options = encoder.categories_[categorical_features_list.index('Occupation')].tolist()
    store_location_options = encoder.categories_[categorical_features_list.index('store_location')].tolist()
except Exception as e:
    st.error(f"Error loading encoder categories: {e}")
    st.stop()


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
        'Season': [selected_season], # Wrap in list for DataFrame
        'Gender': [selected_gender],
        'Age': [age],
        'Occupation': [selected_occupation],
        'Total_Bill': [total_bill],
        'store_location': [selected_store_location]
    }

    # Convert user input to a pandas DataFrame
    input_df = pd.DataFrame(user_input)

    # --- Preprocessing user input ---
    # Separate categorical and numerical columns in the input
    input_categorical_df = input_df[categorical_features_list]
    input_numerical_df = input_df[['Age', 'Total_Bill']]

    # Apply one-hot encoding to categorical features using the loaded encoder
    input_categorical_encoded = encoder.transform(input_categorical_df)

    # Convert the encoded categorical features back to a DataFrame
    # Get the feature names from the encoder after transformation
    encoded_feature_names = encoder.get_feature_names_out(categorical_features_list)
    input_categorical_encoded_df = pd.DataFrame(input_categorical_encoded, columns=encoded_feature_names)

    # Scale numerical features using the loaded scaler
    input_numerical_scaled_df = pd.DataFrame(scaler_classification.transform(input_numerical_df), columns=['Age', 'Total_Bill'])


    # Concatenate the scaled numerical and encoded categorical features
    processed_input_df = pd.concat([input_numerical_scaled_df, input_categorical_encoded_df], axis=1)

    # Ensure the columns in the processed input DataFrame match the training data (X_train_classification)
    # It's best to load X_train_classification's columns or save them during training
    # For this example, let's assume we have the list of training columns available or can derive them
    # In a real application, you might save X_train_classification.columns to a file
    # and load it here to ensure consistency.

    # For now, let's try to construct the expected columns based on preprocessing steps
    # This part needs to be robust and exactly match how X_train_classification was created
    # from the original dataframe.
    # Assuming the order is numerical columns first, then one-hot encoded categorical
    # and drop_first=True was used for one-hot encoding in training

    # Let's load the training columns if they were saved, or reconstruct them accurately
    # If you saved X_train_classification.columns, load it like:
    # training_columns = joblib.load('training_columns.pkl')
    # For now, let's use the columns from the processed_input_df and hope they match
    # This is risky and should be replaced by loading actual training columns if possible.
    training_columns = processed_input_df.columns.tolist() # This is a placeholder, replace with actual training columns

    # Ensure the processed_input_df has all and only the training columns in the correct order
    # Add missing columns with a value of 0
    for col in training_columns:
        if col not in processed_input_df.columns:
            processed_input_df[col] = 0

    # Drop extra columns that are not in training data
    extra_cols = [col for col in processed_input_df.columns if col not in training_columns]
    processed_input_df = processed_input_df.drop(columns=extra_cols)

    # Reindex to ensure the order of columns is the same as the training data
    processed_input_df = processed_input_df[training_columns]


    # --- Make prediction ---
    try:
        prediction = gb_model_clf.predict(processed_input_df)

        # Display the prediction
        st.subheader("Predicted Product Name:")
        st.success(prediction[0])
    except Exception as e:
        st.error(f"Error during prediction: {e}")

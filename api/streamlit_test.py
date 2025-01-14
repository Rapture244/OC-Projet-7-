import streamlit as st
import requests

# Streamlit app title
st.title("Flask API Tester")

# Input for the user to enter an ID
user_id = st.text_input("Enter ID number to predict:", "")

# Button to send the request
if st.button("Get Prediction"):
    # Define the API URL
    url = "http://127.0.0.1:5000/predict"

    # Create the payload
    payload = {"id": int(user_id) if user_id.isdigit() else None}

    # Send the POST request
    if payload["id"] is not None:
        response = requests.post(url, json=payload)

        # Process and display the response
        if response.status_code == 200:
            data = response.json()
            st.success(f"Prediction Successful for ID: {data['SK_ID_CURR']}")
            st.write(f"**Probability:** {data['predicted_proba']}")
            st.write(f"**TARGET:** {data['predicted_target']}")
            st.write(f"**Loan Status:** {data['status']}")
        elif response.status_code == 404:
            st.error(f"Not Found: {response.json()['error']}")
        elif response.status_code == 400:
            st.warning(f"Bad Request: {response.json()['error']}")
        else:
            st.error("An unexpected error occurred.")
    else:
        st.error("Please enter a valid numeric ID.")

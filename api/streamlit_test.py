import streamlit as st
import requests

# Streamlit app title
st.title("Flask API Tester")

# Input for the user to enter an ID
user_id = st.text_input("Enter ID number to predict:", "")

# Button to send the request
if st.button("Get Prediction"):
    # Validate user input
    if not user_id.strip():
        st.error("Please enter a valid ID.")
    elif not user_id.isdigit():
        st.error("ID must be a numeric value.")
    else:
        # Define the API URL
        url = "http://127.0.0.1:5000/predict"

        # Create the payload
        payload = {"id": int(user_id)}

        # Send the POST request
        try:
            response = requests.post(url, json=payload)

            # Check if response is valid JSON
            if response.headers.get("Content-Type") == "application/json":
                data = response.json()

                # Handle response based on status code
                if response.status_code == 200:
                    st.success(f"Prediction Successful for ID: {data['SK_ID_CURR']}")
                    st.write(f"**Probability:** {data['predicted_proba']}")
                    st.write(f"**TARGET:** {data['predicted_target']}")
                    st.write(f"**Loan Status:** {data['status']}")
                elif response.status_code == 404:
                    st.error(f"Error: {data.get('error', 'ID not found in the dataset')}")
                elif response.status_code == 400:
                    st.warning(f"Error: {data.get('error', 'Invalid request payload')}")
                else:
                    st.error(f"Unexpected Error: {data.get('error', 'Unknown error')}")
            else:
                st.error("The API did not return a valid JSON response. Check the API logs.")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

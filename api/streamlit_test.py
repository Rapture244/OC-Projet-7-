"""
This Streamlit application provides a simple user interface for testing the Flask API's `/predict` endpoint.

Key Features:
1. Allows users to input an ID number for prediction.
2. Sends a POST request to the Flask API with the provided ID.
3. Displays the API's response, including predicted probability, target, and loan status.

Functionality:
- `st.title`: Displays the application title.
- `st.text_input`: Captures user input for the ID.
- `st.button`: Triggers the prediction request.
- API Request Handling:
  - Validates user input to ensure it is non-empty and numeric.
  - Sends a POST request to the API with the ID as JSON payload.
  - Handles various API response codes and displays the appropriate messages to the user.
- Exception Handling:
  - Catches and displays errors for network-related issues or invalid API responses.

Dependencies:
- Streamlit: For building the web application.
- Requests: For making HTTP requests to the Flask API.

Notes:
- The API URL (`http://127.0.0.1:5000/predict`) assumes the Flask app is running locally on port 5000. Update this if the API is hosted elsewhere.
- Ensure the Flask API is running and accessible before using this application.
- The application assumes the API returns JSON responses for all valid and error scenarios.

"""

# ====================================================== IMPORTS ===================================================== #
import streamlit as st
import requests


# ==================================================================================================================== #
#                                                       STREAMLIT                                                      #
# ==================================================================================================================== #
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

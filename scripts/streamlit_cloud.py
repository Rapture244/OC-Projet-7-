# ====================================================== IMPORTS ===================================================== #
# Third-party library imports
import requests
import streamlit as st
from loguru import logger
import sys


# ==================================================================================================================== #
#                                                       STREAMLIT APP                                                   #
# ==================================================================================================================== #

# Streamlit app title
st.title("Credit Score Prediction")

# Input for the user to enter an ID
user_id = st.text_input("Enter Credit ID (SK_ID_CURR):", "")

# Button to send the request
if st.button("Get Prediction"):
    # Validate user input
    if not user_id.strip():
        st.error("Please enter a valid ID.")
        logger.warning("User submitted an empty ID.")
    elif not user_id.isdigit():
        st.error("ID must be a numeric value.")
        logger.warning(f"User entered a non-numeric ID: {user_id}")
    else:
        # Define the API URL (Heroku-hosted)
        url = "https://credit-score-attribution-003464da4de3.herokuapp.com/predict"

        # Create the payload
        payload = {"id": int(user_id)}

        # Define the headers
        headers = {
            "Content-Type": "application/json"
            }

        # Log the request details
        logger.debug(f"Sending POST request to {url} with payload: {payload}")

        # Send the POST request
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            logger.debug(f"Received response: {response.status_code} - {response.text}")
        except requests.exceptions.HTTPError as http_err:
            st.error(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
            logger.error(f"HTTP error occurred: {http_err} - Status Code: {response.status_code}")
            if response.headers.get("Content-Type") == "application/json":
                logger.debug(f"Response Content: {response.json()}")
            else:
                logger.debug(f"Response Content: {response.text}")
            st.stop()
        except requests.exceptions.RequestException as err:
            st.error(f"Request failed: {err}")
            logger.error(f"Request exception occurred: {err}")
            st.stop()

        # Process the response
        try:
            data = response.json()
            logger.info("Prediction Successful:")
            logger.info(f"SK_ID_CURR: {data['SK_ID_CURR']}")
            logger.info(f"Predicted Proba: {data['predicted_proba']}")
            logger.info(f"Predicted Target: {data['predicted_target']}")
            logger.info(f"Status: {data['status']}")

            # Display the results in Streamlit
            st.success(f"Prediction Successful for ID: {data['SK_ID_CURR']}")
            st.write(f"**Probability:** {data['predicted_proba']}")
            st.write(f"**TARGET:** {data['predicted_target']}")
            st.write(f"**Loan Status:** {data['status']}")
        except ValueError:
            st.error("Response content is not valid JSON.")
            logger.error("Failed to decode JSON from response.")
            logger.debug(f"Response Content: {response.text}")
        except KeyError as key_err:
            st.error(f"Missing expected key in the response: {key_err}")
            logger.error(f"Missing expected key in the response: {key_err}")
            logger.debug(f"Response JSON: {data}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"An unexpected error occurred while processing the response: {e}")
            logger.debug(f"Response Content: {response.text}")

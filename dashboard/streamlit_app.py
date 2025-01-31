# ==================================================================================================================== #
#                                                        IMPORTS                                                       #
# ==================================================================================================================== #

# Standard library imports
import io
from typing import Optional, Dict, Any, List

# Third-party library imports
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd


# ==================================================================================================================== #
#                                         INITIALIZE SESSION STATE VARIABLES                                          #
# ==================================================================================================================== #
for key in ["cached_client_id", "client_data", "prediction_data", "local_feat_importance", "local_feat_plot", "positioning_plot", "feature_names", "feature_plot"]:
    if key not in st.session_state:
        st.session_state[key] = None  # Set a default value


# ==================================================================================================================== #
#                                                     CONFIGURATION                                                    #
# ==================================================================================================================== #
BASE_URL = "http://127.0.0.1:5000/api"  # Add '/api'

CLIENT_INFO_API_URL = f"{BASE_URL}/client-info"
PREDICT_API_URL = f"{BASE_URL}/predict"
MODEL_PREDICTORS_API_URL = f"{BASE_URL}/model-predictors"
LOCAL_FEATURES_API_URL = f"{BASE_URL}/local-feature-importance"
LOCAL_WATERFALL_PLOT_API_URL = f"{BASE_URL}/local-waterfall-plot"
CLIENT_POSITIONING_PLOT_API_URL = f"{BASE_URL}/client-positioning-plot"
FEATURE_NAMES_API_URL = f"{BASE_URL}/features-name"
FEATURE_POSITIONING_PLOT_API_URL = f"{BASE_URL}/feature-positioning-plot"


# Streamlit App Configuration
PAGE_TITLE = "Credit Score Predictor"
PAGE_ICON = ":bar_chart:"
PAGE_LAYOUT = "wide"


# ====================================================== APP ========================================================= #
# Set the page layout and appearance
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout=PAGE_LAYOUT)

# Add page title explicitly
st.title(PAGE_TITLE)

# ================================================ SET COLORBLIND THEME ============================================== #
# Enable Streamlit's built-in colorblind theme
def set_colorblind_theme():
    st.markdown(
        """<style>
        :root {
            --primary-color: #377eb8;
            --secondary-color: #4daf4a;
            --success-color: #4daf4a;
            --danger-color: #e41a1c;
            --warning-color: #ff7f00;
            --info-color: #984ea3;
            --light-color: #f5f5f5;
            --dark-color: #333333;
        }
        </style>""",
        unsafe_allow_html=True
        )

# Apply Streamlit's built-in colorblind theme
set_colorblind_theme()


# ==================================================================================================================== #
#                                               FUNCTIONS SECTION - BEGIN                                              #
# ==================================================================================================================== #
# =============================================== VALIDATE & FETCH DATA ============================================== #
def validate_and_fetch_data():
    """
    Validates user ID and fetches client & prediction data if ID has changed.
    Uses cached results to avoid unnecessary API calls.
    """
    user_id = st.session_state.get("user_id", "").strip()

    if not user_id or not user_id.isdigit():
        st.sidebar.error("Please enter a valid numeric Client ID.")
        return

    user_id = int(user_id)

    # Check if the client ID has changed
    if st.session_state.get("cached_client_id") == user_id:
        return  # Use cached data, no API call

    # Update cached ID
    st.session_state.cached_client_id = user_id

    # Reset feature plot when new client ID is entered
    st.session_state.feature_plot = None

    # Use cached functions
    st.session_state.client_data = fetch_client_info(user_id)
    st.session_state.prediction_data = fetch_predict_info(user_id)
    st.session_state.local_feat_importance = fetch_local_feat_importance(user_id)
    st.session_state.local_feat_plot = fetch_waterfall_plot(user_id)
    st.session_state.positioning_plot = fetch_positioning_plot(user_id)


# ================================================= FETCH CLIENT INFO ================================================ #
@st.cache_data(show_spinner=False)
def fetch_client_info(user_id: int) -> dict:
    """Fetches client information from API and caches the result."""
    try:
        response = requests.post(url=CLIENT_INFO_API_URL, json={"id": user_id})
        response.raise_for_status()
        return response.json() or {}
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching client data: {e}")
        return {}


# ================================================= FETCH PREDICT INFO ================================================ #
@st.cache_data(show_spinner=False)
def fetch_predict_info(user_id: int) -> dict:
    """Fetches prediction results from API and caches the result."""
    try:
        response = requests.post(url=PREDICT_API_URL, json={"id": user_id})
        response.raise_for_status()
        return response.json() or {}
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching prediction data: {e}")
        return {}



# ================================================= PREDICTION GAUGE ================================================= #
def create_gauge_plotly(predicted_proba: float, threshold: float):
    """
    Create a static gauge chart using Plotly with custom ticks, a gray gauge bar,
    and a pseudo-outline for the gauge. The delta's color changes dynamically:
    green if below the threshold, red if above.

    Args:
        predicted_proba (float): The predicted probability (0 to 1).
        threshold (float): The threshold value (0 to 1).

    Returns:
        plotly.graph_objects.Figure: The gauge plot as a Plotly figure.
    """
    # Define custom tick positions and labels
    custom_ticks = [0, threshold, 1]
    custom_labels = ["0", f"{threshold:.2f}", "1"]

    # Determine the bar color dynamically
    bar_color = "green" if predicted_proba < threshold else "red"

    # Create the gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_proba,
        title={'text': "Default Probability", 'font': {'size': 18}},
        delta={
            "reference": threshold,
            "increasing": {"color": "red"},  # Color when above threshold
            "decreasing": {"color": "green"}  # Color when below threshold
            },
        gauge={
            'axis': {
                'range': [0, 1],
                'tickvals': custom_ticks,
                'ticktext': custom_labels,
                'tickwidth': 1,
                'tickcolor': "darkblue",
                'tickfont': {"size": 18, "color": "black"}  # 🔹 Bigger ticks
                },
            'bar': {'color': bar_color},  # Dynamically set the bar color
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'white'}  # Background effect
                ],
            'threshold': {
                'line': {'color': "darkblue", 'width': 3},
                'thickness': 0.75,
                'value': threshold
                }
            }
        ))

    # Update layout
    fig.update_layout(
        height=275,
        margin={'t': 25, 'b': 25, 'l': 25, 'r': 25},
        paper_bgcolor="white",
        font=dict(color="black", family="Arial")
        )

    return fig


# ================================================= FETCH LOCAL FEATURE IMPORTANCE TABLE ================================================ #
@st.cache_data(show_spinner=False)
def fetch_local_feat_importance(user_id: int) -> list:
    """Fetches local feature importance table from API and caches it."""
    try:
        response = requests.post(url=LOCAL_FEATURES_API_URL, json={"id": user_id})
        response.raise_for_status()
        feature_importance = response.json().get("feature_importance", [])
        return feature_importance
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching local feature importance: {e}")
        return []


# ================================================= FETCH LOCAL FEATURE IMPORTANCE PLOT ================================================= #
@st.cache_data(show_spinner=False)
def fetch_waterfall_plot(user_id: int) -> io.BytesIO:
    """Fetches SHAP waterfall plot from API and caches it."""
    try:
        response = requests.post(url=LOCAL_WATERFALL_PLOT_API_URL, json={"id": user_id})
        if response.status_code != 200:
            st.sidebar.error(f"Error fetching waterfall plot: {response.json().get('error', 'Unknown error')}")
            return None
        return io.BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to fetch waterfall plot: {e}")
        return None


# ============================================== FETCH POSITIONING PLOT ============================================== #
@st.cache_data(show_spinner=False)
def fetch_positioning_plot(user_id: int) -> io.BytesIO:
    """Fetches client positioning plot from API and caches it."""
    try:
        response = requests.post(url=CLIENT_POSITIONING_PLOT_API_URL, json={"id": user_id})
        if response.status_code != 200:
            st.sidebar.error(f"Error fetching positioning plot: {response.json().get('error', 'Unknown error')}")
            return None
        return io.BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to fetch positioning plot: {e}")
        return None

# ================================================= FEATURE NAMES API CALL ================================================= #
@st.cache_data(show_spinner=False)
def fetch_feature_names() -> list:
    """Fetches feature names from the API and caches them."""
    try:
        response = requests.get(FEATURE_NAMES_API_URL)
        response.raise_for_status()
        return response.json().get("features", [])
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Error fetching feature names: {e}")
        return []  # Return an empty list instead of None


# Fetch feature names once when the app starts
if st.session_state.feature_names is None:
    st.session_state.feature_names = fetch_feature_names()


# ================================================= FEATURE POSITIONING API CALL ================================================= #
@st.cache_data(show_spinner=False)
def fetch_feature_positioning_plot(client_id: int, feature_name: str) -> Optional[io.BytesIO]:
    """Fetches feature positioning plot from API and caches it."""
    try:
        response = requests.post(
            url=FEATURE_POSITIONING_PLOT_API_URL,
            json={"id": client_id, "feature": feature_name}
            )
        response.raise_for_status()

        # Return image as BytesIO
        return io.BytesIO(response.content)

    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to fetch feature positioning plot: {e}")
        return None



# ===================================================== MAIN PAGE ==================================================== #
def main_page():
    """
    Renders the main content of the dashboard with styled clickable tabs.
    """
    # Add an information message
    st.info(
        "⭐ This dashboard is designed to assist customer relationship managers in explaining lending decisions transparently. "
        "It enables them to clearly and confidently discuss decisions with customers, reinforcing the company's commitment "
        "to transparency and its core values."
        )

    # ================================================== TABS START HERE ================================================= #
    # Custom CSS for larger clickable tabs, respecting colorblind theme
    st.markdown(
        """
        <style>
        /* Style the tab headers */
        div[class*="stTabs"] button {
            font-size: 18px !important;                         /* Make the text larger */
            padding: 15px 25px;                                 /* Increased padding for bigger tabs */
            background-color: var(--light-color);               /* Use theme's light color */
            color: var(--dark-color);                           /* Use theme's dark color for text */
            border: 1px solid var(--dark-color);                /* Use theme's dark color for border */
            border-radius: 5px;                                 /* Rounded corners for a smoother look */
            margin-right: 5px;                                  /* Add spacing between tabs */
        }

        /* Style for the active/selected tab */
        div[class*="stTabs"] button[aria-selected="true"] {
            background-color: var(--primary-color);             /* Use theme's primary color for active tab */
            color: var(--light-color);                          /* Use theme's light color for text */
            border: 2px solid var(--secondary-color);           /* Use theme's secondary color for active border */
        }

        /* Hover effect for tabs */
        div[class*="stTabs"] button:hover {
            background-color: var(--secondary-color);           /* Use theme's secondary color for hover effect */
            color: var(--light-color);                          /* Use theme's light color for text on hover */
        }
        </style>
        """,
        unsafe_allow_html=True,
        )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Model Predictors (Top 15)",
        "Client Explanatory Variables (Top 15)",
        "Positioning of Credit Applicant in Relation to Other Applications",
        "Positioning of Credit Applicant on a Particular Variable"
        ])

    # --------------------------------------- TAB 1: SHAP Beeswarm Plot (Model Predictors) --------------------------------------- #
    with tab1:
        st.info(
            "ℹ️ This section displays the global feature importance of the model, visualized using SHAP. "
            "⭐ **Scroll down to learn how to read and interpret the plot**."
        )

        # Display the SHAP beeswarm plot (static)
        st.image(
            MODEL_PREDICTORS_API_URL,
            caption="Model Predictors (Top 15)",
            width=900,
        )

        # Explanation for interpreting the SHAP violin plot
        st.markdown("### How to Read and Interpret the SHAP Violin Plot:")
        st.markdown(
            """
            - **Feature Importance**: Each row represents a feature, ranked by its importance to the model's predictions.
              Features at the top have the highest impact.
            - **Color**: The color gradient of the violin plot represents the feature's value (e.g., red for high values, blue for low values).
            - **Violin Shape**:
              - The **width** of the violin at a particular SHAP value represents how many data points (instances) have that SHAP value for the feature.
            - **SHAP Values**:
              - **Positive SHAP Values**: The feature increases the predicted probability of the target class (e.g., loan approval).
              - **Negative SHAP Values**: The feature decreases the predicted probability of the target class.
            - **Takeaway**: The violin plot helps visualize the overall impact and distribution of each feature, providing insights into how the model makes its predictions.
            """
        )

    # --------------------------------------- TAB 2: SHAP Local Features Importance & Waterfall Plot --------------------------------------- #
    with tab2:
        st.info(
            "ℹ️ This section displays the local feature importance for a specific client, visualized using a SHAP "
            "waterfall plot. ⭐ **Scroll down to learn how to read and interpret the plot**."
            )

        # Fetch Local Feature Importance Data
        local_feat_data = st.session_state.get("local_feat_importance", [])

        if local_feat_data:
            # Reorder keys manually to ensure correct column order
            column_order = ["Features", "Value", "SHAP Value", "Description"]
            ordered_local_feat_data = [{col: row[col] for col in column_order} for row in local_feat_data]

            # Display Feature Importance Table
            st.markdown("### Local Feature Importance Table")
            st.table(ordered_local_feat_data)  # Force correct order
        else:
            st.warning("No feature importance data available.")

        # Fetch and Display SHAP Waterfall Plot
        if st.session_state.local_feat_plot is not None:
            st.markdown("### SHAP Waterfall Plot")
            st.image(
                st.session_state.local_feat_plot,
                caption="Local Feature Importance (Top 15)",
                width=1200,
            )
        else:
            st.warning("No SHAP waterfall plot available.")

        # Explanation for interpreting the SHAP waterfall plot
        st.markdown("### How to Read and Interpret the SHAP Waterfall Plot:")
        st.markdown(
            """
            - **Feature Contributions**: Each bar represents a feature's contribution to the model's prediction for the specific client.
              - **Positive Bars**: Push the model's prediction probability higher.
              - **Negative Bars**: Push the model's prediction probability lower.
            - **Base Value**: The starting point of the plot, representing the average prediction across the dataset.
            - **Final Output Value**: The prediction for the specific client, derived by adding the contributions of all features to the base value.
            - **Colors**:
              - **Red Bars**: Features increasing the predicted probability.
              - **Blue Bars**: Features decreasing the predicted probability.
            - **Takeaway**: The waterfall plot breaks down the prediction into feature-level contributions, helping explain why the model made its prediction for the specific client.
            """
            )

    # --------------------------------------- TAB 3: Client Positioning Plot --------------------------------------- #
    with tab3:
        st.info(
            "ℹ️ This section displays the position of the client in relation to other applications "
            "through histograms and boxplots. ⭐ **Scroll down to learn how to interpret the visualizations**."
            )

        # Fetch & Display positionint plot
        if st.session_state.positioning_plot is not None:
            st.markdown("### Client positioning")
            st.image(
                st.session_state.positioning_plot,
                caption="Client Positioning",
                width=1200,
            )

        # Add explanation for interpreting the client positioning visualization
        st.markdown("### How to Read and Interpret the Client Positioning Visualization:")
        st.markdown(
            """
            - **Histograms**:
              - Show the distribution of values for each feature across all applications.
              - The **golden dashed line** represents the client's value.
              - Analyze where the client lies in relation to the overall distribution.
            - **Boxplots**:
              - Show the spread, median, and outliers of each feature across all applications.
              - The **golden dashed line** represents the client's value.
              - Check if the client is an outlier or within the interquartile range (middle 50% of values).
            - **Takeaway**: Use these visualizations to understand how the client's data compares to others, providing insights into the model's prediction.
            """
            )

    # --------------------------------------- TAB 4: Single Feature Positioning Plot --------------------------------------- #
    with tab4:
        st.info(
            "ℹ️ This section allows you to select a specific feature and view the client's positioning "
            "relative to other applicants using histograms and boxplots. "
            "⭐ **Choose a feature from the dropdown to get started.**"
            )

        # Check if a client ID is entered
        client_id = st.session_state.get("cached_client_id")

        if client_id:
            # If no feature is selected, default to the first feature
            if st.session_state.get("selected_feature") is None:
                st.session_state.selected_feature = st.session_state.feature_names[0]

            # Function to fetch feature plot on selection change
            def update_feature_plot():
                selected_feature = st.session_state["selected_feature"]
                if selected_feature:
                    st.session_state.feature_plot = fetch_feature_positioning_plot(client_id, selected_feature)

            # Dropdown for feature selection with callback
            selected_feature = st.selectbox(
                "Select a Feature:",
                st.session_state.feature_names,
                key="selected_feature",
                index=st.session_state.feature_names.index(st.session_state.selected_feature),  # Default selection
                on_change=update_feature_plot  # Trigger plot update when selection changes
                )

            # Ensure the feature plot is fetched on first load
            if st.session_state.feature_plot is None:
                update_feature_plot()  # Fetch the plot for the first feature automatically

            # Display the positioning plot
            if st.session_state.feature_plot is not None:
                st.markdown("### Feature Positioning Plot")
                st.image(st.session_state.feature_plot, caption=f"Positioning Plot for {selected_feature}", width=1200)
            else:
                st.warning("No feature positioning plot available.")
        else:
            st.warning("Feature names are not loaded. Please check the API.")


        # Explanation for interpreting the feature positioning visualization
        st.markdown("### How to Read and Interpret the Feature Positioning Visualization:")
        st.markdown(
            """
            - **Histograms**:
              - Show the distribution of values for the selected feature across all applications.
              - The **golden dashed line** represents the client's value.
              - Analyze where the client lies in relation to the overall distribution.
            - **Boxplots**:
              - Show the spread, median, and outliers for the selected feature across all applications.
              - The **golden dashed line** represents the client's value.
              - Check if the client is an outlier or within the interquartile range (middle 50% of values).
            - **Takeaway**: This visualization helps you understand how the client's data compares to others, providing insights into the model's prediction.
            """
            )


# ==================================================================================================================== #
#                                               FUNCTIONS SECTION - END                                                #
# ==================================================================================================================== #

# ==================================================================================================================== #
#                                                          UI                                                          #
# ==================================================================================================================== #

# ====================================================== SIDEBAR ===================================================== #
st.sidebar.title("Client Selection")

# Input for Client ID with callback
st.sidebar.text_input(
    "Enter Client ID:",
    key="user_id",
    on_change=validate_and_fetch_data
)

# Button to manually trigger data fetch
if st.sidebar.button("Get Client & Prediction Info"):
    validate_and_fetch_data()


# ===================================================== DISPLAY PREDICTION INFO ================================================= #
prediction_info = st.session_state.get("prediction_data")
if prediction_info is None:
    prediction_info = {}  # Ensure it's a dictionary to avoid 'NoneType' issues

prediction_info = prediction_info.get("data", {})


if prediction_info:
    st.sidebar.divider()
    st.sidebar.markdown("<h1 style='margin-bottom: 10px;'>Prediction Results</h1>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<p><b>Client ID:</b> {prediction_info['SK_ID_CURR']}</p>", unsafe_allow_html=True)

    predicted_proba = prediction_info.get("predicted_proba")
    threshold = prediction_info.get("threshold")

    if predicted_proba is not None and threshold is not None:
        st.sidebar.markdown(f"<p><b>Estimated Default Probability:</b> {int(predicted_proba * 100)}%</p>", unsafe_allow_html=True)
        loan_status = prediction_info["status"]
        status_color = "green" if loan_status == "Granted" else "red"
        st.sidebar.markdown(f"<p style='color:{status_color};'><b>Loan Status:</b> {loan_status}</p>", unsafe_allow_html=True)

        gauge_plot = create_gauge_plotly(predicted_proba, threshold)
        st.sidebar.plotly_chart(gauge_plot, use_container_width=True)

# ===================================================== DISPLAY CLIENT INFO ================================================= #
client_info = st.session_state.get("client_data")

if client_info:
    st.sidebar.divider()
    st.sidebar.markdown("<h1 style='margin-bottom: 10px;'>Client Information</h1>", unsafe_allow_html=True)

    def display_client_section(title, section_data, keys):
        if section_data:
            st.sidebar.markdown(f"<h2 style='margin-bottom: 10px;'>{title}</h2>", unsafe_allow_html=True)
            st.sidebar.markdown("\n".join([f"- **{key}**: {section_data[key]}" for key in keys if key in section_data]), unsafe_allow_html=True)
            st.sidebar.divider()

    display_client_section("Personal Profile", client_info.get("Personal Profile"), ["Age (years)", "Gender", "Family status", "Children", "Family members"])
    display_client_section("Financial Profile", client_info.get("Financial Profile"), ["Income type", "Employment Sector", "Income", "Housing situation", "Owns Car", "Owns Real Estate"])
    display_client_section("Credit Profile", client_info.get("Credit Profile"), ["Contract type", "Credit", "Annuity"])

# ================================================= DISPLAY MAIN PAGE ================================================ #
main_page()
# API 

The goal of this API is to offer a classifier model that will allow the financial company called **“Prêt à dépenser”** to know whether to offer consumer credit to people with little or no loan history. As of today, the API ask you for the consumer ID and returns the predicted probability by the model, the predicted classifiacation and a status "Granted" or "Denied". 

From the [Poetry toml file](../../pyproject.toml) you can find the required libaries for the deployment of the API to Heroku which are :  

```python 
requires-python = ">=3.12,<3.13"

# --- Use an array of strings for dependencies ---
dependencies = [
    # Core & API dependencies
    "numpy (>=1.22.0,<2.1)",            # Numerical computations
    "pandas (>=2.2.3,<3.0.0)",          # Data manipulation
    "matplotlib (>=3.10.0,<4.0.0)",     # Plotting and data visualization
    "seaborn (>=0.13.2,<0.14.0)",       # Statistical data visualization
    "loguru (>=0.7.3,<0.8.0)",          # Logging
    "chardet (>=5.2.0,<6.0.0)",         # Character encoding detection
    "setuptools (>=75.8.0,<76.0.0)",    # Required for packaging; Python 3.12 removed distutils
    "pyopencl (>=2024.3,<2025.0)",      # GPU computation

    # API
    "flask (>=3.1.0,<4.0.0)",           # API framework
    "joblib (>=1.4.2,<2.0.0)",          # Model/scaler loading
    "scikit-learn (>=1.0.1,<1.6.0)",    # Preprocessing and machine learning tools
    "werkzeug (>=3.1.3,<4.0.0)",        # For HTTP exceptions

    # Frontend with streamlit
    "streamlit (>=1.41.1,<2.0.0)",      # Web app UI
    "requests (>=2.32.3,<3.0.0)",       # HTTP requests in Streamlit

    # Unit testing
    "pytest (>=8.3.4,<9.0.0)",          # Testing framework

    # Heroku
    "gunicorn (>=23.0.0,<24.0.0)",      # WSGI server for production

    # Additional ones based on trial & error
    "siphash24 (>=1.7,<2.0)",                  # Secure hash function
    "lightgbm (>=4.5.0,<5.0.0)"               # Gradient boosting framework
]

[tool.poetry]
packages = [
    { include = "prod", from = "src" },
    { include = "api"}
]
```

The files related to the API are : 
- [API](api/local_main.py) &xrarr; The file containing the API
    - [Streamlit test of the API in local](scripts/streamlit_local.py)
    - [Streamlit test of the API in the cloud](scripts/streamlit_cloud.py)
    - [Unit testing of the API](tests/test_local_api.py)
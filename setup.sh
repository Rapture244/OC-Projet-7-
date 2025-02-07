#!/bin/bash

# Start Flask API on port 5000
gunicorn api.local_main:app --bind 0.0.0.0:5000 &

# Start Streamlit with headless mode and session timeout
streamlit run dashboard/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 &

# Use wait -n to let Heroku auto-idle when no processes are active
wait -n

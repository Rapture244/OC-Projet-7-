#!/bin/bash

# Start Flask API on port 5000
gunicorn api.local_main:app --bind 0.0.0.0:5000 &

# Start Streamlit on the Heroku-assigned port
streamlit run dashboard/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 &

# Keep dyno alive until both processes are stopped
wait

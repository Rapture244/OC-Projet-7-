#!/bin/bash

# Start Flask API in the background
gunicorn api.local_main:app &

# Wait for Flask to start before Streamlit runs
wait

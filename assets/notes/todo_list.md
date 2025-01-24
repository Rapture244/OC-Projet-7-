# Projet 8 
 Lets fucking go ! 

**Goal** &xrarr; Create an interactif Dashboard 
- [ ] Visualize the score, its probability (is it far from the threshold?) and the interpretation of this score for each customer in a way that is intelligible to a non-data science expert.
- [ ] Visualize key descriptive information about a customer.
- [ ] Use graphs to compare the main descriptive information about a customer with all customers or with a group of similar customers (using a filter system: for example, a drop-down list of the main variables).
- [ ] Take into account the needs of people with disabilities when creating graphics, covering WCAG accessibility criteria.
- [ ] Deploy the dashboard on a Cloud platform, so that it can be accessed by other users on their workstations.
- [ ] **(Bonus)** Optionally (if you have the time): Make it possible to obtain a refreshed score and probability after entering a modification to one or more customer details, as well as to enter a new customer file to obtain the score and probability.


## Tips 

Use a script to launch both the API and then the streamlit app so that all i need is a single heroku app ! 

```shell
# Inside the Procfile 
web: sh setup.sh

# Inside the setup.sh, something like this : 
#!/bin/sh
# Start Gunicorn for the Flask API
gunicorn --bind 0.0.0.0:$PORT -w 4 api.local_main:app &

# Start Streamlit for the Dashboard
streamlit run dashboard/app.py --server.port $PORT
```

API and Dashboard: Ensure both the API and Streamlit dashboard are configured to run on the same server, or use environment variables to define URLs and ports dynamically.











# ML Flow 

**What is Ml Flow ?** &xrarr; MLflow is a platform to manage the machine learning lifecycle
- Track experiments (parameters, metrics, etc.).
- Save and version your models.
- Store and manage artifacts (saved models, plots, datasets).
- You can think of MLflow as a "notebook on steroids" that records your work automatically.

**What is SQLite ?** &xrarr; SQLite is a lightweight, serverless database. It’s just a file (.db) stored on your computer. MLflow can use this file to store experiment metadata (like names, run details, metrics).


**STEPS:**
1. Set up a directory for ML Flow
   - The SQLite database &xrarr; to store expriment metadata
   - Artifacts &xrarr; Saved Models, plots, other outputs 
2. Configure ML Flow to use SQLite
   - Where to store experiment metadata (SQLite Database)
   - Where to store artifacts 
3. Create & Manage Experiments
    - An experiment is a logical grouping of related runs 
      - One experiment might track all runs related to "Fine-tuning ResNet."
      - Another might track "Hyperparameter tuning for SVM."
4. Log Runs and artifacts 
    - A run represent a single model training session. U can log:
    - Parameters (e.g., learning rate, batch size).
    - Metrics (e.g., accuracy, loss).
    - Artifacts (e.g., model files).
5. View the ML Flow UI 
    - MLflow has a UI to view your experiments. You can launch it from the terminal
    - `mlflow ui --backend-store-uri "sqlite:///C:/Users/KDTB0620/Documents/Study/Open Classrooms/Git Repository/projet7/ml_flow/ml_flow.db" --default-artifact-root "file:///C:/Users/KDTB0620/Documents/Study/Open Classrooms/Git Repository/projet7/ml_flow/artifacts"`
6. Save Models in registry for model 
    - Ml Flow can save models in its own format for easier deployment 
    - The model will be saved in the artifacts folder and can be loaded later for evaluation or deployment.
    - In a recent desire of ML Flow to be more free, we moved from stages to now using aliases and tags 


```text
ml_flow/                      # Base directory for MLflow
├── ml_flow.db                # SQLite database for experiment metadata
└── artifacts/                # Directory to store saved models, plots, and other outputs
```

## What is an Experiment in ML Flow ?

An **experiment** is a logical grouping of related **runs**
- *One experiment might track all runs related to "Fine-tuning ResNet."*
- *Another might track "Hyperparameter tuning for SVM."*

Experiments help you:
1. Organize your runs logically.
2. Easily compare different runs within the same experiment.
3. Manage experiment-level metadata (e.g., name, description, lifecycle).

**Key Concepts**
1. **Runs**
- A **run** represents one execution of your code (e.g., training a model with specific hyperparameters).
- Runs are associated with an experiment.
- Each run logs parameters, metrics, and artifacts.

2. **Experiment Metadata**
- Stored in the **SQLite database** (`ml_flow.db`).
- Includes:
    - Experiment **ID** (a unique identifier).
    - Experiment **name** (e.g., "ResNet Fine-Tuning").
    - **Artifact location** (where files like models or plots are saved).
    - **Lifecycle stage** (e.g., `active` or `deleted`).

3. **Lifecycle Stages**
- Experiments have two lifecycle stages:
    - **Active**: The experiment is visible and can be used for logging runs.
    - **Deleted**: The experiment is archived (soft-deleted). It’s not shown in the UI but still exists in the database.



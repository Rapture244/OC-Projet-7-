# Projet 7: Implémentez un modèle de scoring 

## Context 

Vous êtes Data Scientist au sein d'une société financière, nommée **"Prêt à dépenser"**, qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite **mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité** qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un **algorithme de classification** en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

>[!NOTE]  
>Voici [les données](https://www.kaggle.com/c/home-credit-default-risk/data) dont vous aurez besoin pour réaliser l’algorithme de classification. Pour plus de simplicité, vous pouvez les télécharger à [cette adresse](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip).
>
>:bulb: Vous aurez besoin de joindre les différentes tables entre elles.

**Votre mission :**
1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

2. Analyser les features qui contribuent le plus au modèle, d’une manière générale (feature importance globale) et au niveau d’un client (feature importance locale), afin, dans un soucis de transparence, de permettre à un chargé d’études de mieux comprendre le score attribué par le modèle.

3. Mettre en production le modèle de scoring de prédiction à l’aide d’une API et réaliser une interface de test de cette API.

4. Mettre en œuvre une approche globale MLOps de bout en bout, du tracking des expérimentations à l’analyse en production du data drift.

Michaël, votre manager, vous incite à sélectionner un ou des kernels Kaggle pour vous faciliter l’analyse exploratoire, la préparation des données et le feature engineering nécessaires à l’élaboration du modèle de scoring.

>[!NOTE]  
>Si vous le faites, vous devez analyser ce ou ces kernels et le ou les adapter pour vous assurer qu’il(s) répond(ent) aux besoins de votre mission.
>
>Par exemple vous pouvez vous inspirer des kernels suivants : 
>- [Pour l’analyse exploratoire](https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook)
>- [Pour la préparation des données et le feature engineering :](https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features/script) 
>
>C’est optionnel, mais nous vous encourageons à le faire afin de vous permettre de vous focaliser sur l’élaboration du modèle, son optimisation et sa compréhension.

## Key Files

- [(notebook) EDA & Feature Engineering](./notebooks/01.%20Preprocessing.ipynb) &xrarr; Datasets exploration, merging, feature engineering & filling missing values
- [(notebook) Modelisation of the models](notebooks/02.%20Modelisation.ipynb) &xrarr; Models finetuning, refined tuning for chosen model, global & local feature importance, datadrift of the features 
- [The Data Drift report (html)](assets/html/data_drift_report_raw.html)
- [The API](api/local_main.py) 
  - [Streamlit test of the API in local](scripts/streamlit_local.py)
  - [Streamlit test of the API in the cloud](scripts/streamlit_cloud.py)
  - [Python Scrip test of the API (no UI)](scripts/api_test_no_ui.py)
  - [Unit testing of the API](tests/test_local_api.py)

These are some markdown documents that could be helpful to the reader: 
- [Quick API summary](assets/notes/api_summary) 
- [A quick introductory note on ML Flow](assets/notes/mlflow_doc.md)


## Arborescence 

```text
├── .gitignore             # Specifies files and folders to be ignored by Git
├── .python-version        # HEROKU: Specifies the Python version to use
├── .slugignore            # HEROKU: Specifies files and folders to ignore due to 500MB limit
│
├── api                    # API-related files
│   ├── local_main.py      
│   └── __init__.py        
│
├── Aptfile                # HEROKU: Lists Ubuntu packages to install before Python
│
├── assets                 # Static assets for the project
│   ├── html               
│   ├── images             
│   ├── models             # Saved machine learning models
│   ├── notes              # Markdown documentation and notes
│   └── plots              # Generated plot images
│
├── data                   # Datasets storage        
│   ├── processed          
│   ├── raw                
│   └── zip               
│
├── logs                   # Log files for monitoring
│
├── ml_flow                # MLflow-related files
│   ├── artifacts              
│   └── ml_flow.db             
│
├── notebooks              # Jupyter Notebooks for exploration and analysis
│
├── poetry.lock            # Poetry lock file for dependency management
├── Procfile               # HEROKU: Defines the process types to run the API
├── pyproject.toml         # Poetry configuration with library requirements
├── README.md              # Project overview and documentation
├── scripts                # Utility Python scripts
│
├── src                    # Source code packages developed for the project
│   ├── dev                    # Development utilities and modules
│   └── prod                   # Production-ready modules for HEROKU deployment
│
└── tests                  # Unit and integration tests
```





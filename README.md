# Disaster Response Pipeline Project

### Installation
- Install a recent version of python anaconda distribution. Use python3.
- Additionally install the following 3rd party libraries: pandas, ntlk, sklearn, pickle, flask, plotly
- Clone this repository and follow instructions below

### Project Motivation

The motivation of this project is to build a Machine Learning model that can categorize a message into pre-defined labels. The interaction with this model is provided by a Flask web app.

Note that this web app also contains statistics about the original training data.

### File Description

```
─ LICENSE: the license
─ README.md: the readme file
─ app
   ─ run.py - python code for the flask app
   ─ templates
       ─ go.html: results html page
       ─ master.html: main page of the flask app
─ data
   ─ DisasterResponse.db: disaster response database
   ─ disaster_categories.csv: original category data
   ─ disaster_messages.csv: original messages data
   ─ process_data.py: ETL pipeline script
─ models
    - train_classifier.py: ML pipeline script
─ Prototyping\ Disaster\ Response\ Pipeline.ipynb - prototyping jupyter notebook to show the results of the intermediate steps 
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing, Authors, Acknowledgements
Feel free to use the code here as you like.
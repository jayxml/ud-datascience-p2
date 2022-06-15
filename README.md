# Disaster Response Pipeline Project

### Table of contents
1. [Motivation](#motivation)
2. [Libraries used](#library)
3. [File Descriptions](#files)
4. [Summary of the analysis](#summary)
5. [Licensing, authors, acknowledgements](#licensing)
6. [Screenshots of web app and visuals](#screenshots)

## 1. Motivation <a name="motivation"></a>

The goal of the project is to analyze disaser data from Appen (formerly Figure 8) to build a model that classifies diaster messages. The project includes a web app where an emergency work can enter a new message and get classification results in several categories. The exiciting technical part of the project inlcudes using ETL pipelines, NLP pipelines and Machine Learning pipelines.    

## 2. Libraries used <a name="library"></a>

The following Python libraries are used in this project, 
- numpy
- pandas
- sqlalchemy
- re
- nltk (stopwords, WordNetLemmatizer, work_tokenize)
- sklearn (Pipeline, classification_report, train_test_split, RandomForestClassifier, CountVectorizer, TfidfTransformer, MultiOutputClassifier, GridSearchCV )
- pickle

## 3. Files descriptions <a name="files"></a>

- README.md
  The readme file of the project, introducing the motivation of the project, the libraries used, files included and a summary of the analysis results
  
- process_data.py
  This is the Python script that loads data from csv files, cleans data and saves data to a sqlite database
  
- train_classifier.py
  This is the Python script that loads data from the sqlite database, tokenize text, builds, evaluates and saves the model
  
- run.py
  This is the Python script that runs the flask web app so that user can enter new disaster message to get classification results
  
- disaster_messages.csv
  Source data that contains disaster messages
  
- disaster_categories.csv
  Source data that contains categories of disasters
  
  
## 4. How to run <a name="run"></a>
  
  (1) Enter the following commands in the project's root directory
      
      1-a.  run the ETL pipeine to load, cleans and save the data
      python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
  
      1-b. run the ML pipeline to build, evaluate and save the model
      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
      
  (2) Start the web app by entering the following command in the app's directory
      
      python run.py
      
  (3) Visit the web app by going to http://0.0.0.0:3000/ 
  
## 5. Licensing, authors, acknowledgements <a name="licensing"></a> 
  Must give credit to [Appen](https://appen.com/) (formerly Figure 8) for the data. 
  Many thanks to [UDacity](https://learn.udacity.com/) for the Data Science nanodegree program
  
## 6. Screenshots of web app and visuals <a name="screenshots"></a>  

![message classificaiton](/images/message_classification.png)
![message categories](/images/g1_message_categories.png)
![top 5 categories](/images/g2_top5_categories.png)
![category correlation](/images/g3_category_correlation.png)

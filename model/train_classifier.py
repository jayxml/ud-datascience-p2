# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    '''
    load data from the database table

    INPUT:
        database_filepath: the database table
    OUTPUT:
        X: the messages
        Y: the categories
        category names as the list of features
    '''

    # load data from previously saved database table
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_response', engine)

    # define feature and target variables X and Y
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()


    return X, Y, category_names


def tokenize(text):
    
    '''
    normalize case and remove punctuation

    INPUT:
        text: the text to be tokenized

    OUTPUT:
        tokenized clean text
    '''    

    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    '''
    use ML pipeline to build the model and use grid search to improve the model
    '''

    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'tfidf__use_idf': [True, False],
        'clf__estimator__n_estimators': [10, 25],
        'clf__estimator__min_samples_split': [2, 4]

    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evalue the accuracy of the prediction


    INPUT:
        model: the model used for prediction
        X_test: the testing message data
        y_test: the testing category data (the truth)
        category_names: the available category names

    OUTPUT:
        the classification report about prediction accuracy
    '''

    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f'----------------------{i, column}---------------------------------')
        print(classification_report(list(Y_test.values[:, i]), list(Y_pred[:, i])))


def save_model(model, model_filepath):
    '''
    save the model as a pickle file

    INPUT:
        model: the model used for prediction
        model_filepath: the file path where the model is to be saved
    '''

    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
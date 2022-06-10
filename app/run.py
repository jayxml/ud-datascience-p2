import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # For Graph1, get count of categories and category names
    category_counts = df[df.columns[4:]].sum()
    category_names = list(category_counts.index)
    
    # For Graph 2, get the top 5 categories 
    df_t5cat = df.iloc[:, 4:].sum(axis = 0).sort_values(ascending = False)[:5]
    t5_catnames = list(df_t5cat.index.values)
    t5_catvals = list(df_t5cat.values)
        
    # For Graph 3, compute correlation of category columns, 
    # For the graph, it is better to remove the child_alone column since it only has null values
    df2 = df.drop('child_alone', axis=1)    
    corr = df2.iloc[:,4:].corr(method = 'kendall')
    
    # create visuals
    graphs = [
        # GRAPH 1 -  message distribution by category
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts      
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Counts"
                },
                'xaxis': {
                    'title': "Message Categories"
                },      
                'height': 600,
                'margin': dict(
                    pad = 4,
                    b = 200
                )
            }
        },
        # GRAPH 2 -  Top 5 Message Categories
        {
            'data': [
                Bar(
                     x=t5_catvals,
                     y=t5_catnames,                 
                     orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Top 5 Message Categories',
                'yaxis': {
                'title': "Message Categories"
                },
                'xaxis': {
                'title': "Message Counts"
                },
                'margin': dict(
                    pad = 4,
                    b = 100
                )
            }
        },
         # GRAPH 3 -  Message category correlation
        {
            'data': [
                Heatmap(
                    x = corr.columns,
                    y = corr.columns,
                    z = corr.values.tolist(),
                    colorscale = "Earth"
                )
            ],

            'layout': {
                'title': 'Correlation of Message Categories',
                'yaxis': {
                    'title': "",
                    'tickangle': -30
                },
                'xaxis': {
                    'title': "",
                    'tickangle': 30
                },
                'height': 600
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
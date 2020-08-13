import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
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


engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)


model = joblib.load("../models/classifier.pkl")



@app.route('/')
@app.route('/index')
def index():
    


    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    

    categories =  df[df.columns[4:]]
    cate_counts = (categories.mean()*categories.shape[0]).sort_values(ascending=False)
    cate_names = list(cate_counts.index)
    

    direct_cate = df[df.genre == 'direct']
    direct_cate_counts = (direct_cate.mean()*direct_cate.shape[0]).sort_values(ascending=False)
    direct_cate_names = list(direct_cate_counts.index)
    

    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # category plotting
        {
            'data': [
                Bar(
                    x=cate_names,
                    y=cate_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
            
        },
        # Categories Distribution
        {
            'data': [
                Bar(
                    x=direct_cate_names,
                    y=direct_cate_counts
                )
            ],

            'layout': {
                'title': 'Categories Distribution in Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories in Direct Genre"
                }
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

    # use model to predict classification
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
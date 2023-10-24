import pyodbc
from textblob import TextBlob
from jupyter_dash import JupyterDash
from dash import Dash, dash_table,callback
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import dash.dependencies as dd
import plotly.express as px
from dash.dependencies import Input, Output
import pandas as pd
from dash_extensions import Lottie 
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import plotly.graph_objects as go
import base64
import datetime
import io
from dash import no_update
from dash.dependencies import Input, Output, State


def format_number(num):
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)


def parse_contents_nlp(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    
    prediction_vec=[]

    postive_sum=0
    negetive_sum=0
    nutral_sum=0

    for sentence in df.Comment.values:
        testimonial = TextBlob(sentence)
        sentValue=testimonial.sentiment.polarity

        if (sentValue>0.5):
            postive_sum=postive_sum+1
        elif (sentValue<-0.5):
            negetive_sum=negetive_sum+1
        else:
            nutral_sum=nutral_sum+1

        prediction_vec.append(sentValue)
        
    labels=['Positive','Neutral','Negetive']
    values=[postive_sum,nutral_sum,negetive_sum]

    return values








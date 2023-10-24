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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO
import base64
# import plotly.graph_objects as go
import base64
import datetime
import io
from dash import no_update
from dash.dependencies import Input, Output, State
from dash_core_components import Loading

import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from prophet import Prophet


from src.style import colors, colors2,SIDEBAR_STYLE,CONTENT_STYLE,subHeaderStyle,HeaderStyle,url_like,url_comment,url_share,url_more,options
from src.functions import parse_contents_nlp,format_number

app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR],suppress_callback_exceptions=True)

df = pd.read_csv('Datasets//adidas_video_details.csv')
df_fb = pd.read_csv('Datasets//post_details.csv')

# Convert 'Uploaded Date' to datetime
df['Uploaded Date'] = pd.to_datetime(df['Uploaded Date'])
df['Uploaded Date'] = df['Uploaded Date'].dt.tz_convert(None)

df_fb['Uploaded Date'] = pd.to_datetime(df_fb['Uploaded Date'])


sidebar = html.Div(
    [
        html.P("ADIDAS",style=HeaderStyle),
        html.P("Social Media", style=subHeaderStyle),
        html.P("Dashboard", style=subHeaderStyle),
        html.Hr(style={"color": "#ffffff"}),
        html.P(
            " Analytics",style={"color": "#ffffff"}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Youtube", href="/", active="exact"),
                dbc.NavLink("Facebook", href="/page-1", active="exact"),
                dbc.NavLink("Sentiment Analysis", href="/page-2", active="exact"),
                dbc.NavLink("Prediction", href="/page-3", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return [
            dbc.Container([
                  dbc.Row([
             ###################################column###############                  
                    dbc.Col([
                            dbc.Card([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                 dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_like)),
                                                className="col-md-4",
                                            ),
                                            dbc.Col(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Total LIKES Per Vedio", className="card-title text-center"),
                                                            html.H3(id='total-likes',className="card-title text-center",children={}),
                                                        ]
                                                    ),
                                                    className="col-md-8",
                                             ),
                                        ],
                                        className="g-0 d-flex align-items-center",style={"height": "130px"}
                                        )
                                ],
                            className="mb-3",
                            style={"maxWidth": "540px"},color="#05f55d"
                            )

                     ], width=4),
                    
            ###################################column################
        
                    dbc.Col([

                        dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_comment)),
                                        className="col-md-4",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Total COMMENTS Per Vedio", className="card-title text-center"),
                                                    html.H3(id='total-comments',className="card-title text-center",children={}),

                                                ]
                                            ),
                                    className="col-md-8",
                                    ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                        ],
                        className="mb-3",
                        style={"maxWidth": "540px"},color="warning")
                                    ], width=4),
            ###############################column####################

                    dbc.Col([

                         dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_share,className="mb-1")),
                                        className="col-md-4 mb-1",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Total SHARES Per Post", className="card-title text-center"),
                                                    html.H3(id='total-shares',className="card-title text-center",children={}),
                                                ]
                                            ),
                                            className="col-md-8",
                                        ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                            ],
                        className="mb-2",
                        style={"maxWidth": "540px"},color="danger")
                             ], width=4),
            ###################################column###############

                ],className='mb-2 mt-4'),
                dbc.Row([
             ###################################column###############                  
                    dbc.Col([
                            dbc.Card([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                 dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_like)),
                                                className="col-md-4",
                                            ),
                                            dbc.Col(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Average LIKES Per Vedio", className="card-title text-center"),
                                                            html.H3(id='average-likes',className="card-title text-center",children={}),
                                                        ]
                                                    ),
                                                    className="col-md-8",
                                             ),
                                        ],
                                        className="g-0 d-flex align-items-center",style={"height": "130px"}
                                        )
                                ],
                            className="mb-3",
                            style={"maxWidth": "540px"},color="#05f55d"
                            )

                     ], width=4),
                    
            ###################################column################
        
                    dbc.Col([

                        dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_comment)),
                                        className="col-md-4",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Average COMMENTS Per Vedio", className="card-title text-center"),
                                                    html.H3(id='average-comments',className="card-title text-center",children={}),

                                                ]
                                            ),
                                    className="col-md-8",
                                    ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                        ],
                        className="mb-3",
                        style={"maxWidth": "540px"},color="warning")
                                    ], width=4),
            ###############################column####################

                    dbc.Col([

                         dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_share,className="mb-1")),
                                        className="col-md-4 mb-1",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Average SHARES Per Vedio", className="card-title text-center"),
                                                    html.H3(id='average-shares',className="card-title text-center",children={}),
                                                ]
                                            ),
                                            className="col-md-8",
                                        ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                            ],
                        className="mb-2",
                        style={"maxWidth": "540px"},color="danger")
                             ], width=4),
            ###################################column###############

                ],className='mb-2 mt-4'),

               html.Div([
                    html.H3(children='Evolution of Engagement Metrics Over Time', style={'textAlign':'left'}),
                    html.H6(children='Select The Date Range', style={'textAlign':'left'}),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=df['Uploaded Date'].min(),
                        end_date=df['Uploaded Date'].max()
                    ),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content')
                    )
                ]),

                html.Div([
                    html.H3(children='Comparative Analysis of Video Engagement Metrics', style={'textAlign':'left'}),
                    html.H5(children='Select The Vedio Title', style={'textAlign':'left'}),
                    dcc.Dropdown(
                        id='video-dropdown',
                        options=[{'label': i, 'value': i} for i in df['Title'].unique()],
                        value=df['Title'].unique()[0]
                    ),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-bar')
                    )
                ]),
                html.Div([
                    html.H3(children='Proportional Distribution of Engagement Metrics for a Specific Video', style={'textAlign':'left'}),
                    html.H5(children='Select The Vedio Title', style={'textAlign':'left'}),
                    dcc.Dropdown(
                        id='video-dropdown-pie',
                        options=[{'label': i, 'value': i} for i in df['Title'].unique()],
                        value=df['Title'].unique()[0]
                    ),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-pie')
                    )
                ]),
                html.Div([
                    html.H3(children='Correlation Analysis of Video Metrics: Views vs Likes and Comments vs Shares', style={'textAlign':'left'}),
                    html.H5(children='Select The Vedio Title', style={'textAlign':'left'}),
                   
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-scatter')
                    )
                ]),
                html.Div([
                    html.H3(children='Distribution Analysis of Video Durations', style={'textAlign':'left'}),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-hist')
                    )
                ]),
                html.Div([
                    html.H3(children='Visual Representation of Keyword Frequency in Video Content', style={'textAlign':'left'}),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=html.Div(id='graph-content-world')
                    )

                ]),
                
                ])
            ]
    elif pathname == "/page-1":
        return [
            
            dbc.Container([

                  dbc.Row([
             ###################################column###############                  
                    dbc.Col([
                            dbc.Card([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                 dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_like)),
                                                className="col-md-4",
                                            ),
                                            dbc.Col(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Total LIKES Per Post", className="card-title text-center"),
                                                            html.H3(id='total-likes-fb',className="card-title text-center",children={}),
                                                        ]
                                                    ),
                                                    className="col-md-8",
                                             ),
                                        ],
                                        className="g-0 d-flex align-items-center",style={"height": "130px"}
                                        )
                                ],
                            className="mb-3",
                            style={"maxWidth": "540px"},color="#05f55d"
                            )

                     ], width=4),
                    
            ###################################column################
        
                    dbc.Col([

                        dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_comment)),
                                        className="col-md-4",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Total COMMENTS Per Post", className="card-title text-center"),
                                                    html.H3(id='total-comments-fb',className="card-title text-center",children={}),

                                                ]
                                            ),
                                    className="col-md-8",
                                    ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                        ],
                        className="mb-3",
                        style={"maxWidth": "540px"},color="warning")
                                    ], width=4),
            ###############################column####################

                    dbc.Col([

                         dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_share,className="mb-1")),
                                        className="col-md-4 mb-1",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Total SHARES Per Post", className="card-title text-center"),
                                                    html.H3(id='total-shares-fb',className="card-title text-center",children={}),
                                                ]
                                            ),
                                            className="col-md-8",
                                        ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                            ],
                        className="mb-2",
                        style={"maxWidth": "540px"},color="danger")
                             ], width=4),
            ###################################column###############

                ],className='mb-2 mt-4'),
                dbc.Row([
             ###################################column###############                  
                    dbc.Col([
                            dbc.Card([
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                 dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_like)),
                                                className="col-md-4",
                                            ),
                                            dbc.Col(
                                                    dbc.CardBody(
                                                        [
                                                            html.H6("Average LIKES Per Post", className="card-title text-center"),
                                                            html.H3(id='average-likes-fb',className="card-title text-center",children={}),
                                                        ]
                                                    ),
                                                    className="col-md-8",
                                             ),
                                        ],
                                        className="g-0 d-flex align-items-center",style={"height": "130px"}
                                        )
                                ],
                            className="mb-3",
                            style={"maxWidth": "540px"},color="#05f55d"
                            )

                     ], width=4),
                    
            ###################################column################
        
                    dbc.Col([

                        dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_comment)),
                                        className="col-md-4",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Average COMMENTS Per Post", className="card-title text-center"),
                                                    html.H3(id='average-comments-fb',className="card-title text-center",children={}),

                                                ]
                                            ),
                                    className="col-md-8",
                                    ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                        ],
                        className="mb-3",
                        style={"maxWidth": "540px"},color="warning")
                                    ], width=4),
            ###############################column####################

                    dbc.Col([

                         dbc.Card([
                            dbc.Row(
                                [
                                    dbc.Col(
                                         dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_share,className="mb-1")),
                                        className="col-md-4 mb-1",
                                    ),
                                    dbc.Col(
                                            dbc.CardBody(
                                                [
                                                    html.H6("Average SHARES Per Post", className="card-title text-center"),
                                                    html.H3(id='average-shares-fb',className="card-title text-center",children={}),
                                                ]
                                            ),
                                            className="col-md-8",
                                        ),
                                ],
                                className="g-0 d-flex align-items-center",style={"height": "130px"}
                            )
                            ],
                        className="mb-2",
                        style={"maxWidth": "540px"},color="danger")
                             ], width=4),
            ###################################column###############

                ],className='mb-2 mt-4'),

              html.Div([
                    html.H3(children='Evolution of Engagement Metrics Over Time', style={'textAlign':'left'}),
                    html.H5(children='Select The Post Title', style={'textAlign':'left'}),
                    dcc.DatePickerRange(
                        id='my-date-picker-range',
                        min_date_allowed=df['Uploaded Date'].min(),
                        max_date_allowed=df['Uploaded Date'].max(),
                        initial_visible_month=df['Uploaded Date'].min(),
                        start_date=df['Uploaded Date'].min(),
                        end_date=df['Uploaded Date'].max()
                    ),
                    dcc.Graph(id='time-series-graph')
                ]),

                html.Div([
                    html.H3(children='Comparative Analysis of Post Engagement Metrics', style={'textAlign':'left'}),
                    html.H5(children='Select The Post ID', style={'textAlign':'left'}),
                    dcc.Dropdown(
                        id='post-dropdown',
                        options=[{'label': i, 'value': i} for i in df_fb['PostID'].unique()],
                        value=df_fb['PostID'].unique()[0]
                    ),
                    dcc.Graph(id='bar-chart')
                ]),
                html.Div([
                    html.H3(children='Proportional Distribution of Engagement Metrics for a Specific Post', style={'textAlign':'left'}),
                    html.H5(children='Select The Post ID', style={'textAlign':'left'}),
                    dcc.Dropdown(
                        id='video-dropdown-pie-fb',
                        options=[{'label': i, 'value': i} for i in df_fb['PostID'].unique()],
                        value=df_fb['PostID'].unique()[0]
                    ),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-pie-fb')
                    )
                ]),
                html.Div([
                    html.H3(children='Correlation Analysis of Post Metrics:Likes and Comments vs Shares', style={'textAlign':'left'}),
                    html.H5(children='Select The Post ID', style={'textAlign':'left'}),
                   
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=dcc.Graph(id='graph-content-scatter-fb')
                    )
                ]),
                
                html.Div([
                    html.H3(children='Visual Representation of Keyword Frequency in Post Titles', style={'textAlign':'left'}),
                    dcc.Loading(
                        id="loading",
                        type="default",
                        children=html.Div(id='graph-content-world-fb')
                    )

                ]),
                

            ])
               
            
            
            
        ]

    elif pathname == "/page-2":
        return [
            
             dbc.Container([
                  dbc.Row([
                           dcc.Upload(
                            id='upload-data_nlp',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                        ),
                    
                            
                            html.Div(id='output-datatable_nlp'),
                            html.Div(id='output-div_nlp'),
                      
                            

                    ],className='mb-2 mt-4'),
   
                ])
 
        ]
    elif pathname == "/page-3":
        return [
                dbc.Container([



                    html.Div([
                        html.H5(children='Select The Option', style={'textAlign':'left'}),
                        dcc.Dropdown(
                            id='dropdown-pred',
                            options=[
                                {'label': 'Views', 'value': 'Views'},
                                {'label': 'Likes', 'value': 'Likes'}
                            ],
                            value='Views'
                        ),
                        html.H5(children='Select The Date Range', style={'textAlign':'left'}),
                        dcc.Dropdown(
                            id='forecasting_days',
                            options=[{'label': str(i), 'value': i} for i in range(1, 100)],
                            value=30
                        ),
                        dcc.Graph(id='graph-pred')
                    ])

                   
                    
                ])

        ]
    
    # If the user tries to reach a different page, return a 404 message
    return dbc.Jumbotron(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )

@app.callback(
    Output('graph-content', 'figure'),
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    if start_date is None or end_date is None:
        raise PreventUpdate

    # Filter data based on date range
    mask = (df['Uploaded Date'] >= start_date) & (df['Uploaded Date'] <= end_date)
    dff = df.loc[mask]

    # Create traces for each metric
    traces = []
    for metric in ['Views', 'Likes', 'Dislikes', 'Comments', 'Shares']:
        traces.append(go.Scatter(
            x=dff['Uploaded Date'],
            y=dff[metric],
            mode='lines',
            name=metric
        ))

    # Define layout
    layout = go.Layout(
        title='Time Series Plot',
        xaxis=dict(title='Uploaded Date'),
        yaxis=dict(title='Count'),
        hovermode='closest',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgb(255,255,255)')
    )

    return {'data': traces, 'layout': layout}


@app.callback(
    Output('graph-content-bar', 'figure'),
    [Input('video-dropdown', 'value')]
)
def update_graph(selected_video):
    if selected_video is None:
        raise PreventUpdate

    # Filter data based on selected video
    dff = df[df['Title'] == selected_video]

    # Create data for bar chart
    data = [dff['Likes'].values[0], dff['Dislikes'].values[0], dff['Comments'].values[0], dff['Shares'].values[0]]
    labels = ['Likes', 'Dislikes', 'Comments', 'Shares']

    # Create bar chart
    figure = go.Figure(data=[go.Bar(x=labels, y=data)])

    # Add dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })

    return figure

@app.callback(
    Output('graph-content-pie', 'figure'),
    [Input('video-dropdown-pie', 'value')]
)
def update_graph(selected_video):
    if selected_video is None:
        raise PreventUpdate

    # Filter data based on selected video
    dff = df[df['Title'] == selected_video]

    # Create data for pie chart
    data = [dff['Likes'].values[0], dff['Dislikes'].values[0], dff['Comments'].values[0]]
    labels = ['Likes', 'Dislikes', 'Comments']

    # Create pie chart
    figure = go.Figure(data=[go.Pie(labels=labels, values=data)])

    # Add dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })

    return figure

@app.callback(
    Output('graph-content-scatter', 'figure'),
    [Input('url', 'pathname')]
)
def update_graph(pathname):
    # Create scatter plot for 'Views' vs. 'Likes'
    scatter1 = go.Scatter(
        x=df['Views'],
        y=df['Likes'],
        mode='markers',
        name='Views vs Likes'
    )

    # Create scatter plot for 'Comments' vs. 'Shares'
    scatter2 = go.Scatter(
        x=df['Comments'],
        y=df['Shares'],
        mode='markers',
        name='Comments vs Shares'
    )

    # Define layout
    layout = go.Layout(
        title='Scatter Plot',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        hovermode='closest',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgb(255,255,255)')
    )

    return {'data': [scatter1, scatter2], 'layout': layout}


@app.callback(
    Output('graph-content-hist', 'figure'),
    [Input('url', 'pathname')]
)
def update_graph(pathname):

    # Create histogram
    figure = go.Figure(data=[go.Histogram(x=df['Duration'])])

    # Define layout
    figure.update_layout(
        title_text='Histogram of Duration',
        xaxis_title_text='Duration (seconds)',
        yaxis_title_text='Count',
        bargap=0.2,
        bargroupgap=0.1
    )

    # Add dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })

    return figure

@app.callback(
    Output('graph-content-world', 'children'),
    [Input('url', 'pathname')]
)
def update_graph(pathname):
    # Handle Null values and split the keywords
    keywords = df['Keywords'].dropna().str.split(',').sum()
    # Generate word cloud
    wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(keywords))

    # Display the generated image
    fig = plt.figure(figsize=(20,10)) # Adjust the figure size here
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0) # This will help to minimize the white space

    # Convert plot to PNG image
    png_image = io.BytesIO()
    plt.savefig(png_image, format='png', bbox_inches='tight') # bbox_inches='tight' removes extra white spaces
    png_image.seek(0)
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.read()).decode()

    # Create a Dash figure with the image
    img = html.Img(src=png_image_b64_string, style={'width':'100%', 'height':'100%'}) # Adjust the image size here
    return img

@app.callback(
    Output('total-likes', 'children'),
    [Input('url', 'pathname')]
)
def update_total_likes(pathname):
    total_likes = df['Likes'].sum()
    return format_number(total_likes)

@app.callback(
    Output('total-dislikes', 'children'),
    [Input('url', 'pathname')]
)
def update_total_dislikes(pathname):
    total_dislikes = df['Dislikes'].sum()
    return total_dislikes

@app.callback(
    Output('total-comments', 'children'),
    [Input('url', 'pathname')]
)
def update_total_comments(pathname):
    total_comments = df['Comments'].sum()
    return format_number(total_comments)

@app.callback(
    Output('total-shares', 'children'),
    [Input('url', 'pathname')]
)
def update_total_shares(pathname):
    total_shares = df['Shares'].sum()
    return format_number(total_shares)

@app.callback(
    Output('average-likes', 'children'),
    [Input('url', 'pathname')]
)
def update_average_likes(pathname):
    average_likes = df['Likes'].mean()
    return f"{average_likes:.2f}"

@app.callback(
    Output('average-dislikes', 'children'),
    [Input('url', 'pathname')]
)
def update_average_dislikes(pathname):
    average_dislikes = df['Dislikes'].mean()
    return f"{average_dislikes:.2f}"

@app.callback(
    Output('average-comments', 'children'),
    [Input('url', 'pathname')]
)
def update_average_comments(pathname):
    average_comments = df['Comments'].mean()
    return f"{average_comments:.2f}"

@app.callback(
    Output('average-shares', 'children'),
    [Input('url', 'pathname')]
)
def update_average_shares(pathname):
    average_shares = df['Shares'].mean()
    return f"{average_shares:.2f}"

#############facebook##################
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')]
)
def update_graph(start_date, end_date):
    dff = df[(df['Uploaded Date'] >= start_date) & (df['Uploaded Date'] <= end_date)]
    
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=dff['Uploaded Date'], y=dff['Likes'], mode='lines', name='Likes'))
    figure.add_trace(go.Scatter(x=dff['Uploaded Date'], y=dff['Comments'], mode='lines', name='Comments'))
    figure.add_trace(go.Scatter(x=dff['Uploaded Date'], y=dff['Shares'], mode='lines', name='Shares'))

    # Update layout for dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font_color': 'white'
    })

    return figure

@app.callback(
    Output('graph-content-bar-fb', 'figure'),
    [Input('video-dropdown-fb', 'value')]
)
def update_graph(selected_video):
    if selected_video is None:
        raise PreventUpdate

    # Filter data based on selected video
    dff = df_fb[df_fb['Title'] == selected_video]

    # Create data for bar chart
    data = [dff['Likes'].values, dff['Comments'].values, dff['Shares'].values]
    labels = ['Likes', 'Comments', 'Shares']

    # Create bar chart
    figure = go.Figure(data=[go.Bar(x=labels, y=data)])

    # Add dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })

    return figure

@app.callback(
    Output('bar-chart', 'figure'),
    [Input('post-dropdown', 'value')]
)
def update_bar_chart(selected_post):
    
    filtered_df = df_fb[df_fb['PostID'] == selected_post]
    if filtered_df.empty:
        print("empty")
        return go.Figure()  # Returns an empty figure
        # Create a new DataFrame for the bar chart
    chart_df = pd.DataFrame({
        'Interaction': ['Likes', 'Comments', 'Shares'],
        'Count': [filtered_df['Likes'].values[0], filtered_df['Comments'].values[0], filtered_df['Shares'].values[0]]
    })

    fig = px.bar(chart_df, x='Interaction', y='Count')

    # Add dark theme
    # Add dark theme
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })
    

    return fig


@app.callback(
    Output('graph-content-pie-fb', 'figure'),
    [Input('video-dropdown-pie-fb', 'value')]
)
def update_graph(selected_post):
    if selected_post is None:
        raise PreventUpdate

    # Filter data based on selected video
    dff = df_fb[df_fb['PostID'] == selected_post]

    # Create data for pie chart
    data = [dff['Likes'].values[0], dff['Comments'].values[0],dff['Shares'].values[0]]
    labels = ['Likes', 'Comments','Shares']

    # Create pie chart
    figure = go.Figure(data=[go.Pie(labels=labels, values=data)])

    # Add dark theme
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'font': {
            'color': 'white'
        }
    })

    return figure

@app.callback(
    Output('graph-content-scatter-fb', 'figure'),
    [Input('url', 'pathname')]
)
def update_graph(pathname):
    # Create scatter plot for 'Views' vs. 'Likes'
    scatter1 = go.Scatter(
        x=df_fb['Likes'],
        y=df_fb['Comments'],
        mode='markers',
        name='Comments vs Likes'
    )

    # Create scatter plot for 'Comments' vs. 'Shares'
    scatter2 = go.Scatter(
        x=df_fb['Comments'],
        y=df_fb['Shares'],
        mode='markers',
        name='Comments vs Shares'
    )

    # Define layout
    layout = go.Layout(
        title='Scatter Plot',
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        hovermode='closest',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='rgb(255,255,255)')
    )

    return {'data': [scatter1, scatter2], 'layout': layout}


@app.callback(
    Output('graph-content-world-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_graph(pathname):
    # Handle Null values and split the keywords
    # Combine all titles into one large text
    text = ' '.join(title for title in df_fb['Title'])
    # Generate word cloud
    wordcloud = WordCloud(width = 1000, height = 500).generate(text)

    # Display the generated image
    fig = plt.figure(figsize=(20,10)) # Adjust the figure size here
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0) # This will help to minimize the white space

    # Convert plot to PNG image
    png_image = io.BytesIO()
    plt.savefig(png_image, format='png', bbox_inches='tight') # bbox_inches='tight' removes extra white spaces
    png_image.seek(0)
    png_image_b64_string = "data:image/png;base64,"
    png_image_b64_string += base64.b64encode(png_image.read()).decode()

    # Create a Dash figure with the image
    img = html.Img(src=png_image_b64_string, style={'width':'100%', 'height':'100%'}) # Adjust the image size here
    return img


@app.callback(
    Output('total-likes-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_total_likes(pathname):
    total_likes = df_fb['Likes'].sum()
    return format_number(total_likes)

@app.callback(
    Output('total-comments-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_total_comments(pathname):
    total_comments = df_fb['Comments'].sum()
    return format_number(total_comments)

@app.callback(
    Output('total-shares-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_total_shares(pathname):
    total_shares = df_fb['Shares'].sum()
    return format_number(total_shares)

@app.callback(
    Output('average-likes-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_average_likes(pathname):
    average_likes = df_fb['Likes'].mean()
    return f"{average_likes:.2f}"


@app.callback(
    Output('average-comments-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_average_comments(pathname):
    average_comments = df_fb['Comments'].mean()
    return f"{average_comments:.2f}"

@app.callback(
    Output('average-shares-fb', 'children'),
    [Input('url', 'pathname')]
)
def update_average_shares(pathname):
    average_shares = df_fb['Shares'].mean()
    return f"{average_shares:.2f}"


@app.callback(
              Output('output-div_nlp', 'children'),
              Input('upload-data_nlp', 'contents'),
              State('upload-data_nlp', 'filename'),
              State('upload-data_nlp', 'last_modified'))
def update_output_nlp(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:

        list= [parse_contents_nlp(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        labels=['Positive','Neutral','Negetive']
        values=[list[0][0],list[0][1],list[0][2]]

        fig = px.bar(x=labels, y=values)
        fig.update_yaxes(title_text="Counts",showgrid=True, gridwidth=1, gridcolor='#6694cc')
        fig.update_xaxes(title_text="Sentiment Types",showgrid=True, gridwidth=1, gridcolor='#6694cc')


        layout=html.Div([
                       dbc.Row([
            dbc.Col([
                dbc.Card([
                         dbc.CardHeader("Sentiment Analysis",className='bg-info font-weight-bold text-center',style={'font-size':25}),
                        dbc.CardBody([

                            dcc.Graph(figure=fig,style={ 'height':'80vh'} ),
                            
                            

                        ])
                    ],className='rounded-lg border border-light',color="#2C3333"),
                
                
            ],width=12)

            ],className='mb-2 mt-4'),
        ])

        return 
    

# Define callback to update graph
@app.callback(
    Output('graph-pred', 'figure'),
    [Input('dropdown-pred', 'value'), Input('forecasting_days', 'value')]
)
def update_graph(selected_dropdown_value, forecasting_days):
    # Prepare dataframe for Prophet model
    prophet_df = df[['Uploaded Date', selected_dropdown_value]]
    prophet_df.columns = ['ds', 'y']

    prophet_df = prophet_df.sort_values('ds')

    print(prophet_df)

    # Initialize and fit the model
    model = Prophet()
    model.fit(prophet_df)

    # Make future dataframe for specified forecasting period
    future = model.make_future_dataframe(periods=forecasting_days)
    
    # Predict and plot
    forecast = model.predict(future)
        # Create a new figure and add trace
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))

    # Update layout for dark theme
    figure.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    )
    
    return figure


if __name__ == "__main__":
    app.run_server(debug=True,port=7002)

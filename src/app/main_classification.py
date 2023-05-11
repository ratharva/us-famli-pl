import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

mount_point = "./"

csv_path = 'test_output/classification/extract_frames_blind_sweeps_c1_30082022_wscores_test_w16_kmeans/epoch=5-val_loss=0.53/extract_frames_blind_sweeps_c1_30082022_wscores_test_w16_kmeans_sample_test_prediction.parquet'
test_df = pd.read_parquet(csv_path)
batch_size = 256

@app.callback(
    Output('batch-img', 'figure'),
    Input('batch-img', 'figure'),
    Input('pred-class', 'value'))
def batch_img(fig, pred_class):

    fig = go.Figure()

    df_batch = test_df.query('pred_cluster == {pred_class}'.format(pred_class=pred_class))
    df_batch = df_batch.sample(n=batch_size)

    # for idx, row in df_batch.iterrows():
    #     img_path = row['']
    #     img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)

    return fig


app.layout = html.Div(children=[
    dbc.Row(html.H1(children='Classification image visualization app')),
    dbc.Row([dcc.Slider(0, 35, 1, value=0, id='pred-class')]),
    dbc.Row([dcc.Graph(id='batch-img')])
])

if __name__ == '__main__':
    app.run_server(debug=True)
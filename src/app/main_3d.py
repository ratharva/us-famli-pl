import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim
import dash_vtk
from dash_vtk.utils import to_mesh_state


import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.figure_factory as ff

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle
import SimpleITK as sitk

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

mount_point = "./"

csv_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=67-val_loss=8.34/extract_frames_test_sample.csv"
test_df = pd.read_csv(csv_path)

features_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=67-val_loss=8.34/extract_frames_test_sample.pickle"

with open(features_path, 'rb') as f:
    features = pickle.load(f)


# @app.callback(
#     Output('studies-img', 'figure'),
#     Input('studies-img', 'figure'))
# def studies_img(fig):
#     if fig is None:

#         # fig = go.Figure()
#         # fig.add_trace(go.Scatter(x=test_df["pca_0"], y=test_df["pca_1"], mode='markers', showlegend=False,
#         #     marker=dict(color=test_df["pred"], colorscale='sunset', showscale=True))
#         # )
#         fig = ff.create_dendrogram(features)
#         print(features.shape)
#     else:
#         fig = go.Figure(fig)

#     fig.update_layout(autosize=True)
#     return fig


# @app.callback(
#     Output('study-index', 'children'),
#     Output('study-id', 'children'),
#     Output('study-img', 'figure'),    
#     Input('studies-img', 'clickData'),
#     Input('img-size', 'value'))
# def update_img(dict_points, size):
    
#     fig_img = go.Figure()
#     img_path = ""
#     idx = -1
#     if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:
        
#         idx = dict_points["points"][0]["pointIndex"]
        
#         img_path = os.path.join(mount_point, test_df.loc[idx]["img_path"])

#         img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)
        
#         fig_img.add_trace(go.Heatmap(z=np.flip(img_np, axis=0), colorscale='gray'))

#         fig_img.update_layout(
#             autosize=False,
#             width=size,
#             height=size
#         )

#     return ["idx: " + str(idx), "path: " + img_path, fig_img]


xyz = np.array(list(zip(test_df["pca_0"], test_df["pca_1"], test_df["pca_2"])))
labels = test_df["pred"]

vtk_view = dash_vtk.View([
    dash_vtk.PointCloudRepresentation(
        xyz=xyz.ravel(),
        scalars=labels,
        colorDataRange=[np.min(labels), np.max(labels)],
        property={"pointSize": 2},
    ),
])


app.layout = html.Div(children=[
    html.H1(children='MOCO 3D Web App'),
    html.Div([
        html.Div([
            html.Div(
                [html.Div(vtk_view, style={"height": "100%", "width": "100%"}, id='studies-img')],
                className='six columns'
            ),
            html.Div(
                [
                    html.Div([
                        html.Div(html.H2('', id='study-index'), className='two columns'),
                        html.Div(html.H2('id:', id='study-id'), className='ten columns')                    
                    ], className='row'),
                    dcc.Graph(id='study-img'),
                    dcc.Slider(0, 1500, 10, value=450, id='img-size', marks={ 0: {'label': '0'}, 1500: {'label': '1500'}})
                ],
                className='six columns'
                )
        ], className='row')
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
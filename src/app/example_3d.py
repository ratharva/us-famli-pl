import os

import dash
import dash_vtk
from dash import Dash, html, dcc

import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd
import pickle
import SimpleITK as sitk 

from scipy.spatial import KDTree

np.random.seed(42)

mount_point = "./"
csv_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=67-val_loss=8.34/extract_frames_test_sample.csv"
test_df = pd.read_csv(csv_path)

features_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=67-val_loss=8.34/extract_frames_test_sample.pickle"

with open(features_path, 'rb') as f:
    features = pickle.load(f)


points = np.array(list(zip(test_df["pca_0"], test_df["pca_1"], test_df["pca_2"])))

tree = KDTree(points)

xyz = points.ravel()

scalars = test_df["pred"]
min_label = np.amin(scalars)
max_label = np.amax(scalars)
print(f"Number of points: {points.shape}")
print(f"Labels range: [{min_label}, {max_label}]")


# Setup VTK rendering of PointCloud
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

vtk_view = dash_vtk.View(
    id="vtk-view",
    pickingModes=["hover"],
    children=[
        dash_vtk.PointCloudRepresentation(
            xyz=xyz,
            scalars=scalars,
            colorDataRange=[min_label, max_label],
            property={"pointSize": 5},
        ),
         dash_vtk.GeometryRepresentation(
            id="pick-rep",
            actor={"visibility": False},
            children=[
                dash_vtk.Algorithm(
                    id="pick-sphere",
                    vtkClass="vtkSphereSource",
                    state={"radius": 100},
                )
            ],
        )
    ]
)

app.layout = dbc.Container(
    fluid=True,
    style={"marginTop": "15px", "height": "calc(100vh - 30px)"},
    children=[
        html.H1(children='MOCO 3D Web App'),
        dbc.Row(
            [
                dbc.Col(
                    width=8,
                    children=[
                        html.Div(vtk_view, style={"height": "100%", "width": "100%"})
                    ],
                ),
                dbc.Col(width=4, children=[
                    html.Div([
                        html.Div(html.H2('', id='study-index')),
                        html.Div(html.H2('id:', id='study-id'))
                    ]),
                    dcc.Graph(id='study-img'),
                    dcc.Slider(0, 1500, 10, value=450, id='img-size', marks={ 0: {'label': '0'}, 1500: {'label': '1500'}})
                    ])
            ],
            style={"height": "100%"},
        )]
)

@app.callback(
    [
        Output('study-img', 'figure'),
        Output("pick-sphere", "state"),
        Output("pick-rep", "actor"),
    ],
    [   
        Input('study-img', 'figure'),
        Input("vtk-view", "clickInfo"),
        Input("vtk-view", "hoverInfo"),
        Input('img-size', 'value'),
    ],
)
def onInfo(fig_img, clickData, hoverData, size):
    info = hoverData if hoverData else clickData
    if fig_img is None:
        fig_img = go.Figure()
    if info:
        # if (
        #     "representationId" in info
        #     and info["representationId"] == "vtk-representation"
        # ):
        #     return (
        #         [json.dumps(info, indent=2)],
        #         {"center": info["worldPosition"]},
        #         {"visibility": True},
        #     )
        # print(info)
        
        idx = tree.query(info['worldPosition'])[1]
        img_path = os.path.join(mount_point, test_df.loc[idx]["img_path"])

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)
        
        fig_img = go.Figure()
        fig_img.add_trace(go.Heatmap(z=np.flip(img_np, axis=0), colorscale='gray'))

        fig_img.update_layout(
            autosize=False,
            width=size,
            height=size
        )

        return fig_img, dash.no_update, dash.no_update
    return fig_img, {}, {"visibility": False}

if __name__ == "__main__":
    app.run_server(debug=True)
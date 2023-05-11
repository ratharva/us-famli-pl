import os
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import pickle
import SimpleITK as sitk

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)


dataset_mount_folder = '/work/jprieto/data/remote/GWH/Groups/FAMLI/Shared/C1_ML_Analysis'


# In[5]:


dataset_dir = dataset_mount_folder
test_df = pd.read_csv(os.path.join(dataset_dir, 'CSV_files', 'uuid_main_test_new_256.csv'))
test_df = test_df.loc[:,["study_id_uuid","uuid_path", "ga_boe"]].reset_index(drop=True)


# In[6]:

result_name = os.path.join(dataset_dir, 'test_results', 'test_result_ga_1007.pickle')
# weights_fn = result_name.replace('.csv', '_attn_weights.pickle')
scores_fn = result_name.replace('.pickle', '_attn_scores.pickle')
features_fn = result_name.replace('.pickle', '_attn_features.pickle')
x_v_features_fn = result_name.replace('.pickle', '_x_v_features.pickle')
pred_frames_fn = result_name.replace('.pickle', '_pred_frames.pickle')


# In[7]:


result_df = pd.read_pickle(result_name)
print(result_df)
# with open(weights_fn, 'rb') as f:
#     weights = pickle.load(f)
with open(scores_fn, 'rb') as f:
    scores = pickle.load(f)
with open(features_fn, 'rb') as f:
    features = pickle.load(f)
with open(x_v_features_fn, 'rb') as f:
    features_x_v = pickle.load(f)
with open(pred_frames_fn, 'rb') as f:
    pred_frames = pickle.load(f)
print("f", len(scores), len(features), len(features_x_v), len(pred_frames))


# In[8]:


features_np = np.reshape(np.array(features), (-1, 128)).astype(float)
print(np.shape(features_np), features_np.dtype)
X_embedded = TSNE(n_components=2, init='pca').fit_transform(features_np)
result_df["tsne_0"] = X_embedded[:,0]
result_df["tsne_1"] = X_embedded[:,1]
result_df["scores_max"] = [np.max(s.reshape(-1), axis=0) for s in scores]
result_df["prediction_abs"] = np.abs(result_df["truth"] - result_df["pred"])


# In[10]:


class ITKImageDatasetByID(Dataset):
    def __init__(self, df, ga_col, mount_point,
                transform=None):
        self.df = df
        self.mount = mount_point
        self.ga_col = ga_col
        self.transform = transform
        
        self.df_group = self.df.groupby('study_id_uuid')
        self.keys = list(self.df_group.groups.keys())

    def __len__(self):
        return len(self.df_group)

    def __getitem__(self, idx):

        study_id = self.keys[idx]
        df_group = self.df_group.get_group(study_id)
        ga = np.unique(df_group[self.ga_col])[0]
        seq_im_array = []
        for idx, row in df_group.iterrows():
            img_path = row['uuid_path']
            try:
                img, header = nrrd.read(os.path.join(self.mount, img_path), index_order='C')
                assert(len(img.shape) == 3)
                assert(img.shape[1] == 256)
                assert(img.shape[2] == 256)
            except:
                print("Error reading cine: " + img_path)
                img = np.zeros(1, 256, 256)
            seq_im_array.append(img)
        img = np.concatenate(seq_im_array, axis=0)
        if self.transform:
            im_array = self.transform(im_array)
            
        return img, np.array([ga]), study_id

test_df = pd.read_csv(os.path.join(dataset_mount_folder, 'CSV_files', 'uuid_main_test_new_256.csv'))
test_df = test_df.loc[:,["study_id_uuid","uuid_path", "ga_boe"]].reset_index(drop=True)

test_data = ITKImageDatasetByID(test_df, mount_point=dataset_mount_folder,
    ga_col="ga_boe")


# In[11]:


pca = PCA(n_components=2)
X_pca = pca.fit_transform(features_np)
result_df["pca_0"] = X_pca[:,0]
result_df["pca_1"] = X_pca[:,1]


stdev_peaks = []
for idx, (s, pred) in enumerate(zip(scores, pred_frames)):
    s = np.array(s).reshape(-1)
    pred = np.array(pred).reshape(-1)
    
    attention_weights = s / np.sum(s)
    attention_weights_scaled = attention_weights / np.max(attention_weights)
    peak_indices, _ = signal.find_peaks(attention_weights_scaled, distance=25, height=0.01)
    
    stdev_peaks.append(np.std(np.take(pred, peak_indices)))

result_df['stdev_peaks'] = stdev_peaks


current_idx = {"idx": idx, "idx_f": 0, "img_np": [], "df_idx": {}}

def update_study(trace, points, selector):
    if points.trace_name == 'trace 0' and len(points.point_inds) > 0:
        print("update_study", points)
        idx = points.point_inds[0]  
        x_feat_idx = np.array(features_x_v[idx]).reshape(-1, 128)
        x_feat_idx_pca = pca.transform(x_feat_idx)
        scores_idx = np.array(scores[idx]).reshape(-1)

        x_feat_idx_rescaled_pca = pca.transform(np.multiply(x_feat_idx,np.reshape(scores_idx, (-1, 1))))

        df_idx = pd.DataFrame({
            "pca_0": x_feat_idx_pca[:,0],
            "pca_1": x_feat_idx_pca[:,1],
            "pca_rescaled_0": x_feat_idx_rescaled_pca[:,0],
            "pca_rescaled_1": x_feat_idx_rescaled_pca[:,1],
            "scores": scores_idx,
            })
        
        attention_weights = df_idx['scores'] / np.sum(df_idx['scores'])
        attention_weights_scaled = attention_weights / np.max(attention_weights)
        peak_indices = signal.find_peaks(attention_weights_scaled, distance=100, height=0.1)[0]
        
        current_idx["idx"] = idx
        current_idx["img_np"] = test_data[idx][0]
        current_idx["df_idx"] = df_idx
        
        with fig.batch_update():

            fig.data[1]['x'] = df_idx["pca_0"]
            fig.data[1]['y'] = df_idx["pca_1"]
            fig.data[1]['text'] = df_idx["scores"]
            fig.data[1].marker.color = df_idx['scores'] 

            fig.data[2]['x'] = np.take(df_idx["pca_0"], peak_indices)
            fig.data[2]['y'] = np.take(df_idx["pca_1"], peak_indices)
            fig.data[2].marker.color = np.take(df_idx['scores'] , peak_indices)

            fig.data[4]['x'] = [result_df.loc[idx]["pca_0"]]
            fig.data[4]['y'] = [result_df.loc[idx]["pca_1"]]

            fig.data[6]['y'] = attention_weights
            fig.data[7]['x'] = peak_indices
            fig.data[7]['y'] = np.take(attention_weights, peak_indices)
    
fig.data[0].on_click(update_study)

def update_img(trace, points, selector):
    if (points.trace_name == 'trace 1' or points.trace_name == 'trace 6' or points.trace_name == 'trace 7') and len(points.point_inds) > 0:
        print('update_img', points)
        idx_f = points.point_inds[0]
        
        if (points.trace_name == 'trace 6' or points.trace_name == 'trace 7'):
            idx_f = int(points.xs[0])

        current_idx["idx_f"] = idx_f
        
        with fig.batch_update():
            fig.data[3]['x'] = [current_idx["df_idx"]["pca_0"][idx_f]]
            fig.data[3]['y'] = [current_idx["df_idx"]["pca_1"][idx_f]]

            fig.data[5]['z'] = np.flip(current_idx["img_np"][idx_f], axis=0)
    
fig.data[1].2(update_img)
fig.data[6].on_click(update_img)
fig.data[7].on_click(update_img)

def update_color(change):
    with fig.batch_update():
        fig.data[0].marker.color = result_df[colorByDropdown.value]
    

colorByDropdown = widgets.Dropdown(
    description='Color By:   ',
    value='scores_max',
    options=['scores_max', 'stdev_peaks', 'truth']
)
colorByDropdown.observe(update_color, names="value")

container = widgets.HBox(children=[colorByDropdown])
widgets.VBox([container,
              fig])









@app.callback(
    Output('studies-img-clusters', 'figure'),
    Input('studies-img-clusters', 'figure'))

def studies_img_cluster(fig):

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features_centers[:,0], y=features_centers[:,1], mode='text',
        showlegend=False, text=[str(t) for t in np.arange(len(features_centers))], textposition="top center")
    )

    return fig



@app.callback(
    Output('studies-img', 'figure'),
    Input('studies-img', 'figure'),
    Input('score-range-slider', 'value'),
    Input('ga-range-slider', 'value'))

def studies_img(fig, score_range, ga_range):

    query = '{score_min:.3f} <= score and score <= {score_max:.3f} and {ga_min:.3f} <= ga_boe and ga_boe <= {ga_max:.3f}'.format(score_min=score_range[0], score_max=score_range[1], ga_min=ga_range[0], ga_max=ga_range[1])

    print(query)
    test_df_filtered = test_df.query(query)
    
    # fig = go.Figure()

    fig = px.scatter(test_df_filtered, x="pca_0", y="pca_1",
                 hover_data=[test_df_filtered.index],
                 color="pred_cluster")

    # 
    fig.add_trace(go.Scatter(x=features_centers[:,0], y=features_centers[:,1], mode='text',
        showlegend=False, text=[str(t) for t in np.arange(len(features_centers))], textposition="top center")
    )

    return fig


@app.callback(
    Output('studies-img', 'figure'),    
    Input('studies-img', 'figure'))
def update_img(fig):
    
    fig = go.FigureWidget(make_subplots(rows=3, cols=2, column_widths=[0.7, 0.3], specs=[[{'colspan': 2}, {}],[{},{}], [{'colspan': 2},{}]]))

    fig.add_trace(go.Scatter(x=result_df["pca_0"], y=result_df["pca_1"], mode='markers', 
                             text=result_df["truth"], 
                             showlegend=False, 
                             marker=dict(color=result_df["scores_max"], size=result_df["prediction_abs"], showscale=True)
                            ), row=1, col=1)
    fig.add_trace(go.Scatter(mode='markers', showlegend=False, marker=dict(showscale=True)), row=2, col=1)

    fig.add_trace(go.Scatter(mode='markers', showlegend=False, 
                             marker=dict(size=8, opacity=0.9, line=dict(
                                 color='red',
                                 width=2
                             ))), row=2, col=1)

    fig.add_trace(go.Scatter(mode='markers', showlegend=False, 
                marker=dict(size=10, color='magenta'
                )), row=2, col=1)

    fig.add_trace(go.Scatter(mode='markers', marker=dict(color='LightSkyBlue', size=10), showlegend=False), row=2, col=1)
    fig.add_trace(go.Heatmap(colorscale="gray", showscale=False), row=2, col=2)


    fig.add_trace(go.Scatter(mode='lines', showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(mode='markers', showlegend=False, 
                marker=dict(size=10, opacity=0.5,
                    line=dict(
                        color='red',
                        width=2
                    )
                )), row=3, col=1)

    fig.update_layout(
        autosize=False,
        width=800,
        height=800
    )

    fig.data[0].marker.showscale = True
    fig.data[0].marker.colorbar.y = 0.85
    fig.data[0].marker.colorbar.len = 0.3

    fig.data[1].marker.showscale = True
    fig.data[1].marker.colorbar.y = .5
    fig.data[1].marker.colorbar.len = 0.3

    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200
    )

    return fig

app.layout = html.Div(children=[
    html.H1(children='GA Analysis - Web App'),
    html.Div([
        html.Div([
            html.Div(
                [
                dcc.RangeSlider(0, 1, 0.01, value=[.1, 1], id='score-range-slider', marks={ 0: {'label': '0'}, 1: {'label': '1'}}, tooltip={"placement": "bottom", "always_visible": True}),
                dcc.Graph(id='studies-img')],
                className='six columns'
            ),
            html.Div(
                [
                    dcc.Graph(id='study-img'),
                    dcc.Slider(0, 1500, 10, value=450, id='img-size', marks={ 0: {'label': '0'}, 1500: {'label': '1500'}}),
                    html.Div([
                        html.Div(html.H3('', id='study-index'), className='two columns'),
                        html.Div(html.H3('id:', id='study-id'), className='ten columns')                    
                    ], className='row'),
                ],
                className='six columns'
                )
        ], className='row')
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True)
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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

mount_point = "./"

# csv_path = os.path.join(mount_point, 'CSV_files/extract_frames_test_even.csv')
# csv_path = os.path.join(mount_point, 'autoencoder_features_output_even/extract_frames_test_even_spectral_autoencoder_effnet_famligpu_20220614_epoch=23-val_loss=0.001.csv')

# csv_path = os.path.join(mount_point, 'autoencoder_features_output_even/autoencoder_effnet_decode_epoch=55-val_loss=0.00_sample.csv')
# csv_path = os.path.join(mount_point, "test_output/contrastive_learning/epoch=188-val_loss=1.05_dim128_sample.csv")

# csv_path = os.path.join(mount_point, "./test_output/extract_frames/classification/efficientnet_b0_biometry_C1_C2_Annotated_Frames_resampled_256_spc075_uuids_study_uuid_epoch_6/extract_frames_test_even_prediction_sample.csv")
# csv_path = "./test_output/contrastive_learning_512_temp0.2/epoch=594-val_loss=0.67_sample.csv"

# csv_path = "./test_output/contrastive_learning/512_temp0.3/extract_frames_test_even_epoch=25-val_loss=0.48_sample.csv"

# csv_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=3-val_loss=8.82/extract_frames_test_sample.csv"

# train_df = pd.read_parquet("CSV_files/extract_frames_blind_sweeps_train_prediction.parquet")
# train_df = train_df.query("tag != 'BPD' and tag != 'HC' and tag != 'AC' and tag != 'CRL' and tag != 'FL'").reset_index(drop=True)

# csv_path = "./test_output/contrastive_learning/extract_frames_50kHead_bs64_lr1e3_temp0.3_emb32_resnet50_mlp/epoch=36-val_loss=8.29/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.csv"
# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps/epoch=17-val_loss=3.11/extract_frames_test_mlp_sample.csv"
# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps/epoch=10-val_loss=2.95/extract_frames_test_mlp_sample.csv"
# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps/epoch=10-val_loss=4.17/extract_frames_test_sample.csv"
# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps_temp0.2/epoch=6-val_loss=7.76/extract_frames_test_sample.csv"

# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps/epoch=10-val_loss=4.17/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.csv"

# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps_bs512_temp0.3_emb32/epoch=113-val_loss=8.12/extract_frames_test_sample.csv"
# csv_path = "./test_output/contrastive_learning/extract_frames_blindsweeps_c1_voluson_bs64_temp0.3_emb32/epoch=16-val_loss=8.03/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.csv"

# csv_path = "test_output/classification/efficientnet_b0_imagenetfeat/extract_frames_blind_sweeps_c1_30082022_st_voluson_test_prediction_sample.csv"


# csv_path = "./test_output/autoencoder/extract_frames_blindsweeps_c1_voluson_bs256_emb128_monai_autoencoder/epoch=31-val_loss=10.36/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_prediction_sample.csv"
# csv_path = "./test_output/autoencoder/extract_frames_blindsweeps_c1_voluson_bs256_emb128_monai_autoencoder/epoch=31-val_loss=10.36/extract_frames_blind_sweeps_test_prediction_sample.csv"

# csv_path = "./test_output/autoencoder/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4/epoch=27-val_loss=0.13/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_prediction_sample.csv"
# csv_path = "./test_output/autoencoder/extract_frames_blind_sweeps_c1_30082022_wscores/epoch=32-val_loss=0.00/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_prediction_sample.csv"
# csv_path = './test_output/autoencoder/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4_k64/epoch=5-val_loss=0.57/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_prediction_sample.csv'
# csv_path = './test_output/autoencoder/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4_k64_resize/epoch=4-val_loss=0.02/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_prediction_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4_simclr_projhead_rr_pad_rc/epoch=23-val_loss=0.92/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4_rr_pad_rc/epoch=27-val_loss=62.95/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.csv'
# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_1e-4_rr_pad_rc/epoch=27-val_loss=62.95/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_1e-1/epoch=21-val_loss=61.57/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_1e-1_test_sample.csv'

# csv_path = './test_output/autoencoder/extract_frames_blindsweeps_c1_voluson_bs256_emb128_monai_autoencoder/epoch=31-val_loss=10.36/extract_frames_blind_sweeps_test_prediction_sample_wscores.csv'

# csv_path = './test_output/autoencoder/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_k196_rr_rs/epoch=28-val_loss=31.42/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_test_prediction_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_simscore_rr_rs/epoch=38-val_loss=137.59/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_1e-1_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_simscore_rr_rs_k160_droplastdim/epoch=42-val_loss=65.56/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_simscore_rr_rs_k160_droplastdim/epoch=42-val_loss=65.56/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_simscore_rr_rs_k128/epoch=53-val_loss=129.01/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_1e-1_test_sample.csv'

# csv_path = 'test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_simscorew_rr_rs_w4/epoch=54-val_loss=72.98/extract_frames_blind_sweeps_c1_30082022_wscores_voluson_st_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w4/epoch=95-val_loss=46.61/extract_frames_blind_sweeps_c1_30082022_wscores_test_sample.csv'

# csv_path = 'test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_noflyto_simscorew_rr_rs_w4/epoch=66-val_loss=84.91/extract_frames_blind_sweeps_c1_30082022_wscores_test_noflyto_sample.csv'
# csv_path = 'test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w4/epoch=95-val_loss=46.61/extract_frames_blind_sweeps_c1_30082022_wscores_test_sample.csv'

# csv_path = 'test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w8/epoch=126-val_loss=69.45/extract_frames_blind_sweeps_c1_30082022_wscores_test_tsne_0.004_perplexity_300_sample.csv'
# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w8/epoch=126-val_loss=69.45/extract_frames_blind_sweeps_c1_30082022_wscores_test_sample.csv'

# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w16/epoch=105-val_loss=102.57/extract_frames_blind_sweeps_c1_30082022_wscores_test_sample.csv'
# test_df = pd.read_csv(csv_path)

# csv_path = './CSV_files/extract_frames_blind_sweeps_c1_30082022_wscores_test_prediction_perplexity_300_sample.parquet'
# test_df = pd.read_parquet(csv_path)

csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w16/epoch=125-val_loss=102.47/extract_frames_blind_sweeps_c1_30082022_wscores_test_perplexity_300_sample.parquet'

grad_cam_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simscorew_rr_rs_w16/epoch=125-val_loss=102.47/grad_cam'

# csv_path = 'test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_simnorth_lighthouse18/epoch=63-val_loss=1.85/extract_frames_blind_sweeps_c1_30082022_wscores_test_perplexity_300_sample.parquet'
# csv_path = './test_output/contrastive_learning/extract_frames_blind_sweeps_c1_30082022_wscores_moco_v2_resnet50/epoch=175-val_loss=8.10/extract_frames_blind_sweeps_c1_30082022_wscores_test_perplexity_300_sample.parquet'
test_df = pd.read_parquet(csv_path)
column_pred_class = 'pred_cluster'

# print(test_df.columns)


ga_boe_range = [np.floor(np.min(test_df['ga_boe'])), np.ceil(np.max(test_df['ga_boe']))]
ga_boe_range_marks = {}

ga_boe_range_marks[ga_boe_range[0]] = str(ga_boe_range[0])
ga_boe_range_marks[ga_boe_range[1]] = str(ga_boe_range[1])


test_df['abs_g'] = np.abs(test_df['pred'] - test_df['ga_boe'])
test_df['abs_e'] = np.abs(test_df['ga_expert'] - test_df['ga_boe'])

test_df_beat = test_df.query('abs_g < abs_e').groupby(column_pred_class).describe()
# test_df_beat = test_df_beat['score'].sort_values('count', ascending=False)
test_df_beat_g = test_df_beat['abs_g'].copy()
test_df_beat_g['cluster'] = test_df_beat.index
test_df_beat_g = test_df_beat_g.sort_values('mean', ascending=True)

test_df_beat_s = test_df_beat['score'].copy()
test_df_beat_s['cluster'] = test_df_beat.index
test_df_beat_s = test_df_beat_s.sort_values('mean', ascending=False)

# features_path = os.path.join(mount_point, "moco_out_features/moco_extract_feat/features.pickle")
# features_path = os.path.join(mount_point, "autoencoder_features_output_even/autoencoder_effnet_famligpu_20220614.pickle")
# features_path = os.path.join(mount_point, "autoencoder_features_output_even/autoencoder_effnet_similarity_famligpu_20220615_epoch=6-val_loss=0.003.pickle")
# features_path = os.path.join(mount_point, "autoencoder_features_output_even/autoencoder_effnet_famligpu_20220614_epoch=23-val_loss=0.001_sample.pickle")
# features_path = os.path.join(mount_point, "autoencoder_features_output_even/autoencoder_effnet_decode_epoch=55-val_loss=0.00_sample.pickle")
# features_path = os.path.join(mount_point, "test_output/contrastive_learning/epoch=188-val_loss=1.05_dim128_sample.pickle")

# features_path = os.path.join(mount_point, "./test_output/extract_frames/classification/efficientnet_b0_biometry_C1_C2_Annotated_Frames_resampled_256_spc075_uuids_study_uuid_epoch_6/extract_frames_test_even_prediction_sample.pickle")

# features_path = "./test_output/contrastive_learning_512_temp0.2/epoch=594-val_loss=0.67_sample.pickle"

# features_path = "./test_output/contrastive_learning/512_temp0.3/extract_frames_test_even_epoch=25-val_loss=0.48_sample.pickle"

# features_path = "./test_output/contrastive_learning/extract_frames_64_temp0.3_emb32/epoch=3-val_loss=8.82/extract_frames_test_sample.pickle"


# features_path = "./test_output/contrastive_learning/extract_frames_50kHead_bs64_lr1e3_temp0.3_emb32_resnet50_mlp/epoch=36-val_loss=8.29/C2_AnnotatedFrames_FullDetails_Measurable_50kSample_sample.pickle"

features_path = csv_path.replace(os.path.splitext(csv_path)[1], ".pickle")

with open(features_path, 'rb') as f:
    features = pickle.load(f)




# features_centers_path = csv_path.replace(os.path.splitext(csv_path)[1], "_centers.pickle")

# with open(features_centers_path, 'rb') as f:
#     features_centers = pickle.load(f)
# print(features_centers)
# print(features_centers.shape)
# split = 0.1
# idx_sample = np.random.choice(np.arange(features.shape[0]), size=int(features.shape[0]*split))

# test_df = test_df.loc[idx_sample].reset_index(drop=True)
# features = features[idx_sample]

# # pca = PCA(n_components=2)
# # pca_fit = pca.fit_transform(features)

# features_embedded = TSNE(n_components=2, learning_rate='auto', init='random', verbose=1, perplexity=250).fit_transform(features)

# test_df["pca_0"] = features_embedded[:,0]
# test_df["pca_1"] = features_embedded[:,1]

# @app.callback(
#     Output('studies-img-clusters', 'figure'),
#     Input('studies-img-clusters', 'figure'))

# def studies_img_cluster(fig):

#     fig = go.Figure()
#     # fig.add_trace(go.Scatter(x=features_centers[:,0], y=features_centers[:,1], mode='text',
#     #     showlegend=False, text=[str(t) for t in np.arange(len(features_centers))], textposition="top center")
#     # )

#     return fig



@app.callback(
    Output('studies-img', 'figure'),
    Input('studies-img', 'figure'),
    Input('score-range-slider', 'value'),
    Input('ga-range-slider', 'value'), 
    Input('colorby-dropdown', 'value'))
    
def studies_img(fig, score_range, ga_range, color):

    query = '{score_min:.3f} <= score and score <= {score_max:.3f} and {ga_min:.3f} <= ga_boe and ga_boe <= {ga_max:.3f}'.format(score_min=score_range[0], score_max=score_range[1], ga_min=ga_range[0], ga_max=ga_range[1])

    print(query)
    test_df_filtered = test_df.query(query)
    
    # fig = go.Figure()

    fig = px.scatter(test_df_filtered, x="tsne_0", y="tsne_1",
                 hover_data=[test_df_filtered.index],
                 color=color)

    # 
    # fig.add_trace(go.Scatter(x=features_centers[:,0], y=features_centers[:,1], mode='text',
    #     showlegend=False, text=[str(t) for t in np.arange(len(features_centers))], textposition="top center")
    # )

    # fig2 = px.scatter(x=features_centers[:,0], y=features_centers[:,1], text=[str(t) for t in np.arange(len(features_centers))])
    # fig.data[0]['text'] = ['s: {:f}'.format(s) for s in test_df_filtered['score']]

    # print(ga_range)
    # fig.data[0]['marker']['opacity'] = (.astype(float)*test_df_filtered["ga_boe"].astype(float))
    # fig.data = fig.data[::-1]
    # fig.update_layout(autosize=True)

    return fig

    # return fig


@app.callback(
    Output('study-index', 'children'),
    Output('study-id', 'children'),
    Output('study-img', 'figure'),    
    Input('studies-img', 'clickData'),
    Input('img-size', 'value'))
def update_img(dict_points, size):
    
    fig_img = go.Figure()
    img_path = ""
    idx = -1
    if dict_points is not None and dict_points["points"] is not None and len(dict_points["points"]) > 0 and dict_points["points"][0]["curveNumber"] == 0:
        
        idx = dict_points["points"][0]["customdata"][0]
        
        img_path = os.path.join(mount_point, test_df.loc[idx]["img_path"])

        img_np = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.ubyte)

        img_path_grad = os.path.join(grad_cam_path, img_path)

        fig_img.add_trace(go.Heatmap(z=np.flip(img_np, axis=0), colorscale='gray'))

        if os.path.exists(img_path_grad):

            img_np_grad = sitk.GetArrayFromImage(sitk.ReadImage(img_path_grad)).astype(np.ubyte)
        
            fig_img.add_trace(go.Heatmap(z=np.flip(img_np_grad, axis=0), colorscale='jet', opacity=0.3, showlegend=False))

        fig_img.update_layout(
            autosize=False,
            width=size,
            height=size
        )

    return ["idx: " + str(idx), "path: " + img_path, fig_img]

# @app.callback(
#     Output('batch-img', 'figure'),
#     Input('studies-img', 'figure'),
#     Input('new-batch', 'n_clicks'),
#     Input('score-min', 'value'),
#     Input('score-max', 'value'),
#     )
# def new_batch(fig, n_clicks, score_min, score_max):
        
#     batch_size = 256

#     img = []    
#     filtered_df = train_df.query("{score_min:.10f} <= score and score <= {score_max:.10f}".format(score_min=score_min, score_max=score_max)).reset_index(drop=True)
#     print(score_min, score_max, len(filtered_df))
#     for i in range(batch_size):
#         img_path = filtered_df.loc[np.random.randint(len(filtered_df))]["img_path"]            
#         img.append(sitk.GetArrayFromImage(sitk.ReadImage(img_path)))
#     img = np.array(img)    
#     fig = px.imshow(img, facet_col=0, binary_string=True, facet_col_wrap=8, facet_row_spacing=0.01, height=8192)        

    

#     # fig.update_layout(autosize=False, 
#     #     width="100%", 
#     #     height="100%")
#     return fig

app.layout = html.Div(children=[
    html.H1(children='SimN - Web Analysis App'),
    html.Div([
        html.Div([
            html.Div(
                [
                dcc.Dropdown([column_pred_class, 'score', 'ga_boe', 'abs_g'], id='colorby-dropdown'),
                dcc.RangeSlider(0, 1, 0.01, value=[.1, 1], id='score-range-slider', marks={ 0: {'label': '0'}, 1: {'label': '1'}}, tooltip={"placement": "bottom", "always_visible": True}),
                dcc.RangeSlider(ga_boe_range[0], ga_boe_range[1], 1, value=[ga_boe_range[0], ga_boe_range[1]], id='ga-range-slider', marks=ga_boe_range_marks, tooltip={"placement": "bottom", "always_visible": True}),
                # dcc.Graph(id='studies-img-clusters'),
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
        ], className='row'),
        html.Div(
            [html.Div(html.H3('abs(GA_BOE - PRED)', id='t-beat_g')),
            dash_table.DataTable(id='table-beat', columns=[{"name": i, "id": i} 
             for i in test_df_beat_g.columns],
                data=test_df_beat_g.to_dict('records'),
                style_cell=dict(textAlign='left'),
                style_header=dict(backgroundColor="paleturquoise"),
                style_data=dict(backgroundColor="lavender")),
            html.Div(html.H3('SCORE', id='t-score')),
            dash_table.DataTable(id='table-beat-score', columns=[{"name": i, "id": i} 
             for i in test_df_beat_s.columns],
                data=test_df_beat_s.to_dict('records'),
                style_cell=dict(textAlign='left'),
                style_header=dict(backgroundColor="paleturquoise"),
                style_data=dict(backgroundColor="lavender"))],
            className='twelve columns'
        )
        # html.Div([
        #     html.Div(
        #         [html.Button('New batch', id='new-batch', n_clicks=0),
        #         dcc.Input(id="score-min", type="number", value=0),
        #         dcc.Input(id="score-max", type="number", value=1)],
        #         className='twelve columns'
        #     ),
        #     html.Div(
        #         [dcc.Graph(id='batch-img')],
        #         className='twelve columns'
        #     )
        # ], className='row')
    ])
])

if __name__ == '__main__':
    app.run_server(debug=True, port=8787)
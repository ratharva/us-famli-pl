import pandas as pd
import os

df = pd.read_csv("CSV_files/Dataset_B2_masked_resampled_256_spc075_frames_prediction.csv")
df_bs = pd.read_csv("CSV_files/B2a_MFM_Sweeps_Resampling_masked_resampled_256_spc075.csv")

df['key'] = df['img_path'].str.replace('.nrrd', '')
df['key'] = df['img_path'].apply(lambda f: os.path.dirname(f))
df['key'] = df['key'].str.replace('extract_frames_blind_sweeps/', '')

df_bs['key'] = df_bs['img_path'].str.replace('.nrrd', '')

df_merged = df.merge(df_bs, how='inner', on='key')
df_merged.rename(columns={'img_path_y': 'img_path_orig', 'img_path_x': 'img_path'}, inplace=True)
df_merged.to_csv("CSV_files/Dataset_B2_masked_resampled_256_spc075_frames_prediction.csv_", index=False)


import os
import pandas as pd
import numpy as np



def compute_ga(g):
	scores = np.array(g['score'])
	ga = np.array(g['pred'])

	w = scores/np.sum(scores)
	ga_study = np.sum(ga*w)

	return ga_study

df = pd.read_csv("CSV_files/Dataset_B2_masked_resampled_256_spc075_frames_prediction.csv")

ga_study = df.groupby('PID_date').apply(compute_ga).to_frame('pred_study')

df = df.merge(ga_study, on='PID_date')

df.to_csv('CSV_files/Dataset_B2_masked_resampled_256_spc075_frames_prediction_studyprediction.csv', index=False)




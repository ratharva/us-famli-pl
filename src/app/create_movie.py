import cv2
import numpy as np
import glob
import argparse
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

def main(args):
    df = pd.read_csv(args.csv)

    if(args.query):
        df = df.query(args.query)    

    img_array = []

    if(args.total is None):
        total = len(df['img_path'])
    else:
        total = args.total
        frac = total/len(df)
        print(frac)
        df = df.sample(frac=frac)

    for img_path in tqdm(df['img_path'], total=total):

        img = sitk.ReadImage(img_path)
        img_np = sitk.GetArrayFromImage(img)
        
        height = img_np.shape[0]
        width = img_np.shape[1]
        size = (width, height)
        img_array.append(img_np)


    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    for i in range(len(img_array)):
        out.write(np.repeat(np.expand_dims(img_array[i], -1), 3, -1))
    out.release()


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Create movie')    
    parser.add_argument('--csv', type=str, help='CSV file with column img_path', required=True)        
    parser.add_argument('--total', type=int, help='Number of frames', default=None)        
    parser.add_argument('--query', type=str, help='Query the dataframe', default=None)        
    parser.add_argument('--out', help='Output filename', type=str, default='out.avi')


    args = parser.parse_args()

    main(args)
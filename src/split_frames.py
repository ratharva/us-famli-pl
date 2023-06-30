import SimpleITK as sitk
import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

def main(args):

    if not os.path.exists(args.out):
        img = sitk.ReadImage(args.img)
        img_np = sitk.GetArrayFromImage(img).astype(args.type)

        if img.GetDimension() == 3: 
    
            os.makedirs(args.out)
        
            for idx, frame_np in tqdm(enumerate(img_np), total=len(img_np)):

                if len(frame_np.shape) == 2:
                    out_img = sitk.GetImageFromArray(frame_np)
                elif len(frame_np.shape) == 3:
                    out_img = sitk.GetImageFromArray(frame_np, isVector=True)
                else:
                    print("4D image?", file=sys.stderr)
                    raise 

                out_img.SetSpacing(img.GetSpacing()[0:2])
                out_img.SetOrigin(img.GetOrigin()[0:2])        

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(args.out, str(idx) + ".nrrd"))
                writer.UseCompressionOn()
                writer.Execute(out_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, help='Input image', required=True)
    parser.add_argument('--out', type=str, help='Output directory', required=True)
    parser.add_argument('--type', type=str, help='Output type', default="ubyte")

    args = parser.parse_args()

    main(args)
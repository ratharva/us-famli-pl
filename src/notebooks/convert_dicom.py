import os
import math 

import numpy as np
from monai.data import ITKReader


import itk

import argparse

def main(args):
    reader = ITKReader()
    img = reader.read(args.dir)
    itk.imwrite(img, args.out)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Convert dicom seq')
    parser.add_argument('--dir', help='Dicom directory', type=str, required=True)
    parser.add_argument('--out', help='Output filename', type=str, required=True)


    args = parser.parse_args()

    main(args)

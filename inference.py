import torchvision
import torch
import torchvision.transforms as transforms

import argparse

import pandas as pd
import numpy as np
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt

from autoencoder import Autoencoder
from annotator import normalize_maxmin, batch_viewer
from preprocessor import Preprocessor

CSV_FILE = "centers.csv"

def plot_hist(bins=32):
    df = pd.read_csv(CSV_FILE)

    plt.figure()
    plt.hist(df["SCORE"], bins=np.arange(0, 1, 1 / bins), color='b', alpha=0.5, label='Output', edgecolor='black', linewidth=1.0)
    plt.ylabel('Frequency')
    plt.xlabel('Probability of having Prostate Cancer')
    plt.xlim(0, 1)
    plt.legend(loc='upper right')
    plt.title('Inference Results')

    plt.show()

    pass

def infer_dir(dir):
    dcm_dirs = {}
    for (root, dirs, file) in os.walk(dir):
        for f in file:
            if '.dcm' in f:
                # print(root, dirs, f)
                # dcm_dirs.append(root)
                if root not in dcm_dirs:
                    dcm_dirs[root] = 1
                else:
                    dcm_dirs[root] += 1

    dirs_to_infer = []
    for k in dcm_dirs:
        if dcm_dirs[k] > 14:
            dirs_to_infer.append(k)

    print(len(dirs_to_infer))
    
    infer(dirs_to_infer)


def infer(dicom_dirs):

    try:
        df = pd.read_csv(CSV_FILE)
        df.set_index("DICOM_DIR", inplace=True)
    except:
        df = pd.DataFrame(columns=["DICOM_DIR", "X", "Y", "Z", "SCORE"])
        df.set_index("DICOM_DIR", inplace=True)
        df.to_csv(CSV_FILE)

    print("Creating Model")
    sizes = [[2, 19, 19], [4, 38, 38], [7, 75, 75], [14, 150, 150]]
    model = Autoencoder(in_channels=1, n_classes=1, sizes=sizes, aug_classes=8, debug=False)

    # model.cuda(0)
    # print("# Sent Model to Cuda ", 0)

    _ = model.load_state_dict(torch.load("model.pt"))
    _ = model.eval()

    print(f"Reading {len(dicom_dirs)} samples")
    for d in dicom_dirs:
        print()
        print("Sample: ", d)

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(d)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()
        img = np.array(sitk.GetArrayFromImage(image)).astype(float)

        # print(img.shape)

        # Reshape to [Height, Width, Depth]
        if img.shape[1] == img.shape[2]: 
            img = np.transpose(img, (1, 2, 0))

        # print(img.shape)
        mx, mn = 500, -250
        # print(mx, mn)
        img = normalize_maxmin(img, mx, mn)
        # print(df.loc[d])
        if d not in df.index:
            x, y, z = batch_viewer(img, d, index=True)
            if x == 0 and y == 0 and z == 0:
                continue
        else:
            row = df.loc[d]
            x, y, z = row["X"], row["Y"], row["Z"]

        prostate_center = (y,x,z)

        # print("Center at: ", prostate_center)

        p = Preprocessor(150, 150, 14, None)
        img = p.localise_input(img, prostate_center)

        # _ = batch_viewer(img, dicom_dir, index=True)

        transform = transforms.Compose([
                #transforms.RandomCrop(64, padding=None),
                transforms.ToTensor(),
            ])

        img = transform(img)
        test_img = img.type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
        # print(test_img.shape)

        _, score = model(test_img, getScores=True)
        print("\tProbability of PCa: ", score.item())
        print()

        if d not in df.index:
            new_row = pd.DataFrame({"DICOM_DIR":d, "X":x, "Y":y, "Z":z, "SCORE":score.item()}, index=["DICOM_DIR"])
            new_row.set_index("DICOM_DIR", inplace=True)
            df = df.append( new_row )
            df.to_csv(CSV_FILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prostate Cancer Inference')
    parser.add_argument('--infer_sample', type=str, default="")
    parser.add_argument('--infer_dir', type=str, default="")
    parser.add_argument('--results', action="store_true")
    args = parser.parse_args()
    # exit()
    if args.infer_sample != "":
        infer(dicom_dirs=[args.infer_sample])
    elif args.infer_dir != "":
        infer_dir(dir=args.infer_dir)
    elif args.results:
        plot_hist()
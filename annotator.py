import pandas as pd
import numpy as np
import os
import SimpleITK as sitk

import matplotlib.pyplot as plt



root_dir = "E:/data/ctcontrast/"
csv_file = "DATA/ct_labels.csv"


################################################################################
def normalize_maxmin(x, mx, mn):
    x = np.clip(x, mn, mx)
    return (x - mn) / (mx - mn)

################################################################################

loc_x, loc_y, loc_z = 0, 0, 0

def batch_viewer(volume, id=0, index=False):
    #print(volume.shape)
    #volume = volume.cpu().numpy()
    volume = np.squeeze(volume)
    #print(volume.shape)
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.ind = index
    if ax.ind:
        ax.index = volume.shape[2] // 2 - 1
        ax.index = 10
        ax.imshow(volume[:, :, ax.index], vmin=0, vmax=1, cmap='gray')
    else:
        ax.index = volume.shape[0] // 2 - 1
        ax.imshow(volume[ax.index, :, :], vmin=0, vmax=1, cmap='gray')
    ax.id = id
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.canvas.mpl_connect('key_press_event', process_key_b)
    fig.canvas.mpl_connect('button_press_event', on_mouse_move)
    plt.show()
    global loc_x
    global loc_y
    global loc_z
    return loc_x, loc_y, loc_z

def on_mouse_move(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    try:
        x = round(event.xdata)
        y = round(event.ydata)
        z = ax.index
        print('Coords: '+str(ax.id)+", "+str(x)+","+str(y)+","+str(z))
        global loc_x
        global loc_y
        global loc_z
        loc_x, loc_y, loc_z = x, y, z
    except Exception as e:
        print(e)
        # print('Coords:', ax.id, ",\"\",\"\",\"\"")
    
def process_key_b(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'x':
        previous_slice_b(ax)
    elif event.key == 'z':
        next_slice_b(ax)
    elif event.key == 'c':
        print_ax(ax)
    fig.canvas.draw()

def previous_slice_b(ax):
    """Go to the previous slice."""
    volume = ax.volume
    if ax.ind:
        ax.index = (ax.index + 1) % volume.shape[2]
        #print(ax.index)
        ax.images[0].set_array(volume[:, :, ax.index])
    else:
        ax.index = (ax.index + 1) % volume.shape[0]
        #print(ax.index)
        ax.images[0].set_array(volume[ax.index, :, :])

def print_ax(ax):
    print(ax.index)

def next_slice_b(ax):
    """Go to the next slice."""
    volume = ax.volume
    if ax.ind:
        ax.index = (ax.index - 1) % volume.shape[2]
        #print(ax.index)
        ax.images[0].set_array(volume[:, :, ax.index])
    else:
        ax.index = (ax.index - 1) % volume.shape[0]
        #print(ax.index)
        ax.images[0].set_array(volume[ax.index, :, :])

################################################################################

def annotate():
    # some calculations for normalizing the data
    mx, mn = 500, -250
    wc, ww = mx + mn // 2, mx - mn
    print(mx, mn, wc, ww)

    # CSV file contain columns -> 
    #   [PID, CT, MRI, TARGET, CT_X, CT_Y, CT_Z]
    data_frame = pd.read_csv(csv_file)

    # Total Size = 580 Scans
    #   146 Positives + 434 Negatives

    # Read only things that have a target and a file location
    data_frame = data_frame[ data_frame["CT"].notna() & data_frame["TARGET"].notna() ]
    # Filter out things that do not already have an X, Y, Z location
    # data_frame = data_frame[ data_frame["CT_X"].isna() & data_frame["CT_Y"].isna() & data_frame["CT_Z"].isna() ]

    print("Read ", len(data_frame), "samples")

    # rename variable, and filter excess columns that we don't need
    data = data_frame[["PID", "CT", "TARGET", "CT_X", "CT_Y", "CT_Z"]]

    for index, sample in data.iterrows():
        # get the file location
        path_input = os.path.join(root_dir, sample["CT"][2:] )

        print("Loading: ", index, "/", len(data) - 1 ,": ", path_input)
        target = sample["TARGET"]
        pid = sample["PID"]

        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path_input)
        reader.SetFileNames(dicom_names)

        image = reader.Execute()
        img = np.array(sitk.GetArrayFromImage(image)).astype(np.float)

        # print(img.shape)

        # Reshape to [Height, Width, Depth]
        if img.shape[1] == img.shape[2]: 
            img = np.transpose(img, (1, 2, 0))

        # print(img.shape)

        img = normalize_maxmin(img, mx, mn)

        batch_viewer(img, pid, index=True)

if __name__ == "__main__":
    annotate()
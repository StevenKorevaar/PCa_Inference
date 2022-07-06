# Prostate Cancer Inference

## Files Included:
- README.md: This file.
- inference.py: The main Python script which will be used to infer samples.
- annotator.py: a secondary script containing functions for displaying and annotating prostate CT scans (finding the centers)
- autoencoder.py: the model used for training and inference, this holds the architecture of the file (altering this file will mean the pretrained model will not be able to be used).
- preprocessor.py: the preprocessor for each sample, takes in a sample and X, Y, Z coordinates of the center of the prostate and crops it to that location.
- requirements.txt: a file containing all the required Python packages to run the inference modules.
- model.pt: the parameters and weights for the pretrained model.

## How to run:
There are three main modes of operation that I've provided:
1. Single sample inference:
2. Directory based inference
3. Results plotting

You will need to install all the required Python modules first using:

```pip install -r requirements.txt```

Or using your preferred package manager.

The only other requirement is to have a somewhat specific folder structure for your data. Each "sample" should be contained in its own folder, so .dcm files per sample are separated. Then each sample's folder should be within another folder. You can have as many subfolders as you like though, the program looks through them all recursively.

At minimum the structure should look something like this:

```
data/
|
|___sample_1/
|       |___1.dcm
|       |___2.dcm
|       ...
|       |___X.dcm
|
|___sample_2/
|       |___1.dcm
|       |___2.dcm
|       ...
|       |___X.dcm
...
|___sample_Y/
        |___1.dcm
        |___2.dcm
        ...
        |___X.dcm
```

### Single sample inference:
To infer for a single sample use the command "--infer_sample" followed by the directory of the .dcm files that you want to be read.
eg: 
    ```python inference.py --infer_sample "D:\data\PCa\2\CT_Contrast_Cancer_2\Anonymized - 100998\Ct Abdo Pelvis With Contrast\Ax 5mm PV Abdo - 2"```

Inside the directory directly should be the sequence of .dcm files. It's worth noting that you need to include the quotation marks around the directory, in case there are spaces; otherwise the argument parser might fail.

The procedure for inference is as follows:

The program will attempt to read a .csv file (centers.csv) stored in the same directory as inference.py, this file stores all the previously inferred samples and their directories. If the .csv file does not exist the program will create one for you.

Then the program will attempt to read `model.pt` and create the pretrained model from the file.

Then the program will loop through all supplied directories (if coming from multi sample inference) or just the one you provided via command line arguments.

It will first read all the dicom files into a single image, and normalize it for viewing along with adjusting axes if necessary.

Then the program will check if you have localized the prostate previously already or not. If it has been done, the program will use the stored values to crop the image to the prostate, otherwise, a figure will pop up showing you the CT scan. You can use the Z and X keys to scroll through all the slices of the scan to find the center of the prostate. Once you have found it, click on it: this will print out a set of coordinates (and the directory) to your console (which you can check if you want to). You can click as many times as you want to to make sure you get it as accurate as possible, only the last one will be used by the program.

Then you can close the figure and the program will take the X, Y, and Z coordinates you last clicked on and crop the scan to 150x150x14 in the X, Y, and Z axes, hopefully getting the whole prostate around it.

If you do not want to use the scan, simply close the figure without clicking on the image, and the program will abort that sample and continue to the next one.

Then the program will put the sample through the model and calculate the probability and output it to the console. Once the probability is calculated the program will save the X, Y, and Z coordinates and the score and put them into `centers.csv`. When you're done inferring, you can run the results mode, to see a histogram of all the scores outputted by the model.

### Multi sample inference:

```python inference.py --infer_dir "D:\data\PCa\2\CT_Contrast_Cancer_2"```

The above command will go through the directory given `"D:\data\PCa\2\CT_Contrast_Cancer_2"` recursively and find all folders which have more than 14 .dcm files within it and then call single sample inference on them one at a time, which should hopefully allow you to go through all your samples relatively quickly.

### Results:
```python inference.py --results```

The above command simply takes all the inferences done (which are stored in centers.csv), and plots them on a histogram. You can alter the number of bins of the histogram as you please by editing the function header (on line 19 of inference.py).
import numpy as np
import random

class Preprocessor:
    '''
    Holds all the functions required to turn each .npz scan/file into usable data for the Prostate AI U-Net.

    Member Variables:
        Height - the height each sample will be cropped to (center + each direction by <height / 2>)
        Width - the Width each sample will be cropped to (center + each direction by <Width / 2>)
        Depth - the number of total slices captured from original scan
        Center_Slice - The approximate location of the prostate from the end of the image
    '''
    height = 0
    width = 0
    depth = 0
    center_slice = 50

    def __init__(self, height, width, depth, center_slice):
        '''
            Height - the height each sample will be cropped to (center + each direction by <height / 2>)
            Width - the Width each sample will be cropped to (center + each direction by <Width / 2>)
            Depth - the number of total slices captured from original scan
            Center_Slice - The approximate location of the prostate from the end of the image
        '''
        self.height = height
        self.width = width
        self.depth = depth
        self.center_slice = center_slice
        pass

    def start_end(self, center, size, max):
        #print(center, size, max)
        start, end = 0, 0

        start = round(center - size / 2)
        if start <= 0:
            start = 0
            end = size

        if end != size:
            #print("here ", end, size)
            end = round(center + size / 2)

        if end >= max:
            end = max
            start = round(max - size)

        return start, end

    def localise_input(self, input, prostate_center=(200,200,50), take_prostate=0):
        '''
            Takes in an input image and the center coordinates of the prostate (x, y, z), and returns the localised input image.
        
            Finds a rough bounding box around the prostate.
        '''
        input_loc = input

        input_width, input_height, input_slices = input.shape
        center_w, center_h, center_d = prostate_center

        # print(input.shape)

        width_start, width_end = self.start_end(center_w, self.width, input_width)
        height_start, height_end = self.start_end(center_h, self.height, input_height)
        input_slice_start, input_slice_end = self.start_end(center_d, self.depth, input_slices)

        #print(width_start, width_end, height_start, height_end, input_slice_start, input_slice_end)
        width_start = int(width_start)
        width_end = int(width_end)

        height_start = int(height_start)
        height_end = int(height_end)

        input_slice_start = int(input_slice_start)
        input_slice_end = int(input_slice_end)

        if take_prostate == 0:
            ws, we = width_start, width_end
            hs, he = height_start, height_end
            ds, de = input_slice_start, input_slice_end
        else:
            overlapping = True
            while overlapping:

                w = random.randint(0, input_width)

                h = random.randint(0, input_height)
                d = random.randint(0, input_slices)

                ws, we = self.start_end(w, self.width, input_width)
                hs, he = self.start_end(h, self.height, input_height)
                ds, de = self.start_end(d, self.depth, input_slices)

                ws = int(ws)
                we = int(we)

                hs = int(hs)
                he = int(he)

                ds = int(ds)
                de = int(de)

                overlapping = self.checkOverlap( 
                    ( ws, we, hs, he, ds, de ),
                    ( width_start, width_end, height_start, height_end, input_slice_start, input_slice_end ), 
                )

        # print(ws, we, hs, he, ds, de)
        input_loc = input[ws:we, hs:he, ds:de]
        return input_loc

    def checkOverlap(self, x, y):
        overlapping = True
        
        x1s, x1e, y1s, y1e, z1s, z1e = x
        x2s, x2e, y2s, y2e, z2s, z2e = y
        
        # print("Testing: ")
        # print(x1s, x1e, y1s, y1e, z1s, z1e)
        # print(x2s, x2e, y2s, y2e, z2s, z2e)

        if x1e < x2s or x1s > x2e: overlapping = False
        elif y1e < y2s or z1s > z2e: overlapping = False
        elif z1e < z2s or z1s > z2e: overlapping = False
        
        return overlapping

def main():
    return

if __name__ == "__main__":
    main()

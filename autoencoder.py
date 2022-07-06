import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import math
import numpy as np

from functools import partial

#from BASE.datasets.CT_Contrast import CTContrastDataSet

################################################################################
# Main Class

class Autoencoder(nn.Module):
    """
        Based heavily on:
        https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c

        args:
        - in_channels
        - n_classes
        - sizes: the list of sizes to upscale to in the encoder, in ascending order
        - debug=False
    """
    def __init__(self, in_channels, n_classes, sizes, aug_classes=8, debug=True, *args, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.debug = debug

        self.channel_sizes = [ 32, 64, 128 ]
        # self.channel_sizes = [ 8, 16, 32 ]
        self.feature_size = self.channel_sizes[-1]

        self.encoder = encoder(self.in_channels, self.channel_sizes, debug=debug)
        self.decoder = decoder(self.channel_sizes, sizes=sizes, debug=debug)
        
        self.feat_extract = nn.AdaptiveAvgPool3d( (1) )

        self.final = nn.Sequential(
            nn.Dropout(p=0.1), 
            nn.Linear(self.feature_size, n_classes),
            # act(),
            # nn.Dropout(p=0.1),
            # nn.Linear(self.feature_size // 2, n_classes),
            # nn.Sigmoid(),
        )

        self.final_aug = nn.Sequential( 
            nn.Linear(self.feature_size, self.feature_size // 2),
            act(),
            nn.Linear(self.feature_size // 2, aug_classes),
            # nn.Sigmoid(),
        )

        self.final_detect = nn.Sequential( 
            nn.Linear(self.feature_size, self.feature_size // 2),
            act(),
            nn.Linear(self.feature_size // 2, 1),
            # nn.Sigmoid(),
        )

        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x, mode='', getFeatures=False, getScores=False):
        """ 
        Passes an input image (x) through the network
        """
        if self.debug: print("Mode: ", mode)
        ssl = False
        aug = False
        detect = False
        if mode == 'ssl': ssl = True
        if mode == 'aug': aug = True
        if mode == 'detect': detect = True

        out = self.encoder(x, ssl=ssl)
        
        if self.debug: print("Feature Layer Size: ", out.shape)

        if ssl:
            # If pretraining: decode the image for reconstruction
            if self.debug: print("out ", out.shape)
            out = self.decoder(out)
            if self.debug: print("Final reconstruction shape ", out.shape)
        else:
            # Otherwise make a classification guess
            out = self.feat_extract(out).squeeze()
            
            if self.debug: print("Features shape: ", out.shape)
            
            # Return aggregated features
            if getFeatures:
                return out

            if aug:
                out = self.final_aug(out)
                if getScores: 
                    scores = self.softmax(out)
                    return out, scores
                return out

            if detect:
                out = self.final_detect(out)
                if getScores: 
                    scores = self.sig(out)
                    return out, scores
                return out
            
            out = self.final(out)

            if getScores: 
                scores = self.sig(out)
                return out, scores
            if self.debug: print("Final classification shape: ", out.shape)
            

        return out

################################################################################
# Helper Classes and Functions

# https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278

class Conv3dAuto(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding =  (self.kernel_size[0] // 2, 
            self.kernel_size[1] // 2, 
            self.kernel_size[2] // 2 ) 

conv3x3x3 = partial(Conv3dAuto, kernel_size=3, bias=False)     


class DeConv3dAuto(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # dynamic add padding based on the kernel_size
        self.padding =  (self.kernel_size[0] // 2, 
            self.kernel_size[1] // 2, 
            self.kernel_size[2] // 2 ) 

deconv3x3x3 = partial(DeConv3dAuto, kernel_size=3, bias=False)  


# Initial momentum=0.4
def bn_act(in_channels, *args, **kwargs):
    return nn.Sequential( nn.BatchNorm3d(in_channels, momentum=0.4), act())

# def bn_act(in_channels, *args, **kwargs):
#     return nn.Sequential( act(), nn.BatchNorm3d(in_channels, momentum=0.4))

# def bn_act(in_channels=1):
#     return nn.Sequential(nn.ReLU(inplace=True))

def act(in_channels=1):
    return nn.Sequential(nn.ReLU(inplace=True))

################################################################################
# Blocks

class encoder(nn.Module):
    def __init__(self, in_channels, channel_sizes, debug=False, *args, **kwargs):
        super().__init__()
        self.debug = debug
        
        self.d0 = Down(in_channels=in_channels, out_channels=channel_sizes[0], debug=debug, stride=2, kernel_size=3, final=True)
        self.d1 = Down(in_channels=channel_sizes[0], out_channels=channel_sizes[1], debug=debug, stride=2, kernel_size=3, final=True)
        self.d2 = Down(in_channels=channel_sizes[1], out_channels=channel_sizes[2], debug=debug, stride=2, kernel_size=3, final=True)
        
        pass

    def forward(self, x, ssl=False):
        if self.debug: print("Starting Encoding")

        out = self.d0(x, ssl=ssl)
        if self.debug: print("out0 ", out.shape)

        out = self.d1(out, ssl=ssl)
        if self.debug: print("out1 ", out.shape)

        out = self.d2(out, ssl=ssl)
        if self.debug: print("out2 ", out.shape)

        return out


class Down(nn.Module):
    def __init__(self, in_channels, out_channels=64, debug=False, stride=2, kernel_size=3, final=False):
        super().__init__()
        self.debug = debug
        self.out_channels = out_channels

        if not final:
            self.layer = nn.Sequential(
                # Channel Addition
                # conv3x3x3(in_channels=(in_channels), out_channels=(in_channels), stride=1, kernel_size=kernel_size),
                conv3x3x3(in_channels=(in_channels), out_channels=(out_channels), stride=stride, kernel_size=kernel_size),
                act(in_channels=(self.out_channels)),
            )
        else:
            self.layer = nn.Sequential(
                # Channel Addition
                # conv3x3x3(in_channels=(in_channels), out_channels=(in_channels), stride=1, kernel_size=kernel_size),
                conv3x3x3(in_channels=(in_channels), out_channels=(out_channels), stride=stride, kernel_size=kernel_size),
                bn_act(in_channels=(self.out_channels)),
            ) 

        # https://towardsdatascience.com/unet-line-by-line-explanation-9b191c76baf5
        # Larger P => More zero'd elements: ie. 0.9 => 90% of elements are turned to zero
        # self.drop = nn.Dropout(p=0.2)

    def forward(self, x, ssl=False):
        out = x
        out = self.layer(out)

        # if ssl: out = self.drop_ssl(out)
        # out = self.drop(out)
        return out


class decoder(nn.Module):
    """
        in_channels
        channel_mult
        sizes
        debug
    """
    def __init__(self, channel_sizes, sizes, debug=False):
        super().__init__()

        self.debug = debug
        self.sizes = sizes
        self.base_size = sizes[-1]

        self.u0 = Up(in_channels=channel_sizes[0], out_channels=1, debug=debug, final=True, stride=2, kernel_size=3)
        self.u1 = Up(in_channels=channel_sizes[1], out_channels=channel_sizes[0], debug=debug, stride=2, kernel_size=3)
        self.u2 = Up(in_channels=channel_sizes[2], out_channels=channel_sizes[1], debug=debug, stride=2, kernel_size=3)

        pass

    def forward(self, x):
        if self.debug: print("Starting Decoding")

        out = self.u2(x, self.sizes[1])
        if self.debug: print("2. ", out.shape)

        out = self.u1(out, self.sizes[2])
        if self.debug: print("1. ", out.shape)

        out = self.u0(out, self.sizes[3])
        if self.debug: print("0. ", out.shape)

        if self.debug: print("out shape ", out.shape)

        return out

class Up(nn.Module):
    """
        in_channels
        channel_mult
        sizes
        debug
    """
    def __init__(self, in_channels=2, out_channels=2, final=False, debug=False, stride=2, kernel_size=3):
        super().__init__()
        self.debug = debug
        self.final = final
        # self.conv = conv3x3x3(in_channels=(in_channels), out_channels=(in_channels), stride=1, kernel_size=kernel_size)
        self.deconv = deconv3x3x3(in_channels=(in_channels), out_channels=(out_channels), stride=stride, kernel_size=kernel_size)
        self.bn2 = act(in_channels=(out_channels))
        if not self.final: self.bn2 = bn_act(in_channels=(out_channels))

    def forward(self, x, output_size):
        if self.debug: print("x ", x.shape)
        # out = self.conv(x)
        out = self.deconv(x, output_size=output_size)
        out = self.bn2(out)
        
        return out


class feature_extractor(nn.Module):
    """
        This class is the feature extractor block of VNet++.

        args:
        - in_channels
        - n_classes
    """
    def __init__(self, in_channels, debug=False):
        super().__init__()
        self.debug = debug

        self.pool = nn.AdaptiveAvgPool3d( (1) )

    def forward(self, x):

        if self.debug: print("classifier in x", x.shape)
        x = self.pool(x).squeeze()
        if self.debug: print("pooled over x", x.shape)
        return x

class feature_extractor_old(nn.Module):
    """
        This class is the feature extractor block of VNet++.

        args:
        - in_channels
        - n_classes
    """
    def __init__(self, in_channels, n_classes, size, debug=False):
        super().__init__()
        self.debug = debug
        if self.debug: print(size)

        self.conv = conv3x3x3(in_channels=in_channels, out_channels=1, stride=1, kernel_size=1)
        self.bn = act()

        linear_channels = np.prod(size[-2:]) # * in_channels

        self.decoder = nn.Sequential (
            nn.Linear(linear_channels, n_classes),
            act(),
            # nn.Sigmoid()
        )

    def forward(self, x):

        if self.debug: print("classifier in x", x.shape)
        x = self.conv(x)
        x = self.bn(x)
        if self.debug: print("channel pooled x", x.shape)
        x = x.mean(2)
        if self.debug: print("meaned over depth x", x.shape)
        x = x.view(x.size(0), -1)
        if self.debug: print("reshaped x", x.shape)
        x = self.decoder(x)
        if self.debug: print("Classifier out: ", x.shape)
        return x

################################################################################
# Testing

def test_autoencoder():

    print("-----------------------------------------")
    print("Testing Autoencoder Model")
    input = Variable(torch.ones(5, 1, 14, 150, 150)).cuda()
    sizes = [[2, 19, 19], [4, 38, 38], [7, 75, 75], [14, 150, 150]]
    model = Autoencoder(in_channels=1, n_classes=1, sizes=sizes, debug=True)
    model.cuda()
    #print(model)
    # summary(model, (1, 14, 150, 150))
    print("-----------------------------------------")
    out = model(input, mode='')
    print("TRAINING OUTPUT: ", out.shape)
    print("-----------------------------------------")
    out = model(input, mode='ssl')
    print("CLASSIFICATION OUTPUT: ", out.shape)

    exit()

if __name__ == '__main__':
    test_autoencoder()

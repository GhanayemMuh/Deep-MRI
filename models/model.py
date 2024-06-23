import torch
import torch.nn as nn
import torch.nn.functional as F
from models.subsampling import SubsamplingLayer

class MyReconstructionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, drop_rate=0.5, device='cpu', learn_mask=False):
        super(MyReconstructionModel, self).__init__()
        
        self.subsample = SubsamplingLayer(drop_rate, device, learn_mask)

        self.enc_conv1 = self.contract_block(in_channels, 64, 3, 1)
        self.enc_conv2 = self.contract_block(64, 128, 3, 1)
        self.enc_conv3 = self.contract_block(128, 256, 3, 1)
        self.enc_conv4 = self.contract_block(256, 512, 3, 1)
        

        self.dec_conv4 = self.expand_block(512, 256, 3, 1)
        self.dec_conv3 = self.expand_block(256, 128, 3, 1)
        self.dec_conv2 = self.expand_block(128, 64, 3, 1)
        self.dec_conv1 = self.expand_block(64, out_channels, 3, 1)

    def forward(self, x):
        x1 = self.subsample(x)

        # Encoder
        e1 = self.enc_conv1(x1)
        e2 = self.enc_conv2(e1)
        e3 = self.enc_conv3(e2)
        e4 = self.enc_conv4(e3)
        
        # Decoder
        d4 = self.dec_conv4(e4)
        d3 = self.dec_conv3(d4 + e3)
        d2 = self.dec_conv2(d3 + e2)
        d1 = self.dec_conv1(d2 + e1).squeeze(1)
        
        return d1,self.subsample(x)

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return contract
    
    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        return expand

import torch
from torch import nn
from torch import Tensor
class WavefunctionUNet(nn.Module):
    def __init__(self,channels:list=None):
        '''
        UNET model designed to predict wavefunctions given their potentials
        :param channels: List containing number of channels per layer
        '''
        super().__init__()
        self.start = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=channels[0],kernel_size=3,padding=1,padding_mode='reflect'),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=channels[0],out_channels=channels[0],kernel_size=3,padding=1,padding_mode='reflect'),
                                   nn.ReLU())
        self.end = nn.Sequential(nn.Conv2d(in_channels=channels[0],out_channels=1,kernel_size=3,padding=1,padding_mode='reflect'),
                                 nn.Tanh(),
                                 )
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(1,len(channels)):
            self.encoders.append(self.make_encoder(channels[i-1],channels[i]))
            self.decoders.append(self.make_decoder(channels[i],channels[i-1],channels[i-1]))
        self.bridge = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channels[-1],out_channels=channels[-1],kernel_size=3,padding=1,padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[-1],out_channels=channels[-1],kernel_size=3,padding=1,padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[-1],out_channels=channels[-1],kernel_size=3,padding=1,padding_mode='reflect'),
            nn.ReLU(),
        )

    def forward(self,X:Tensor)->Tensor:
        '''
        Predicts wavefunction using potential energy
        :param X: Potential Image
        :return: Wavefunction Prediction
        '''

        X = self.start(X)
        encoded_levels = []
        for encoder in self.encoders:
            encoded_levels.append(X)
            X = encoder(X)
        X = self.bridge(X)
        for i in range(len(self.decoders)-1,-1,-1):
            X = self.decoders[i]['decode'](X)
            X = self.decoders[i]['convolution'](torch.cat((X, encoded_levels[i]), dim=1))
        X = self.end(X)
        X = X * torch.sqrt(1/(X**2).sum(dim=(1,2,3),keepdim=True)) #normalization
        return X

    def make_encoder(self,input_size:int,output_size:int)->nn.Sequential:
        '''
        Creates encoding layer of UNET
        :param input_size: Number of input channels
        :param output_size: Number of output channels
        :return: Encoding Layer
        '''
        return nn.Sequential(nn.MaxPool2d(kernel_size=2,stride=2),
                             nn.Conv2d(in_channels=input_size,out_channels=output_size,kernel_size=3,padding=1,padding_mode='reflect'),
                             nn.ReLU(),
                             nn.Conv2d(in_channels=output_size,out_channels=output_size,kernel_size=3,padding=1,padding_mode='reflect'),
                             nn.ReLU())

    def make_decoder(self,input_size:int,output_size:int,skip_size:int)->nn.ModuleDict:
        '''
        Creates decoding layer of UNET
        :param input_size: Number of input channels
        :param output_size: Number of output channels
        :param skip_size: Number of skip connections
        :return: Decoding Layer
        '''
        return nn.ModuleDict({'decode':nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear'),
                                                     nn.Conv2d(in_channels=input_size,out_channels=output_size,kernel_size=3,padding=1,padding_mode='reflect'),
                                                     nn.ReLU()),
                              'convolution':nn.Sequential(nn.Conv2d(in_channels=output_size+skip_size,out_channels=output_size,kernel_size=3,padding=1,padding_mode='reflect'),
                                                          nn.ReLU(),
                                                          nn.Conv2d(in_channels=output_size,out_channels=output_size,kernel_size=3,padding=1,padding_mode='reflect'),
                                                          nn.ReLU(),
                                                          )})
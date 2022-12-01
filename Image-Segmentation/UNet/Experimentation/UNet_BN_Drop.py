import torch.nn as nn
import torch
import numpy as np
from PIL import Image
# import torchvision.transforms as 

def convBox (input_dim,out_dim):
    conv = nn.Sequential(
        nn.Conv2d(input_dim,out_dim,kernel_size=5,stride =1 ,padding=0),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.15)
        
        # nn.Conv2d(out_dim,out_dim,kernel_size= 3,stide=1,padding=1)
    )
    return conv

def deconvBox(input_dim,out_dim):
    deconv = nn.Sequential(
        nn.Conv2d(input_dim,out_dim,kernel_size=5,stride =1, padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )
    return deconv

def crop_tensor(x,y):
    x_size = x.size()[2]
    y_size = y.size()[2]
    delta = x_size-y_size
    delta = delta//2
    x = x[:,:,delta:x_size-delta,delta:x_size-delta]
    return x
class UNet_BN_Drop(nn.Module):
    def __init__(self):
        super(UNet_BN_Drop,self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride=2)
        self.conv1 = convBox(3,64)
        self.conv2 = convBox(64,128)
        self.conv3 = convBox(128,256)
        self.conv4 = convBox(256,512)
        self.conv5 = convBox(512,1024)
        self.upsample1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.deconv1 = deconvBox(1024,512)
        self.upsample2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.deconv2 = deconvBox(512,256)
        self.upsample3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.deconv3 = deconvBox(256,128)
        self.upsample4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.deconv4 = deconvBox(128,64)
        self.out = nn.Conv2d(64,1,stride=1, kernel_size =3)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x1 = self.conv1(x)
        print(x1.size(),'x1')
        x2= self.maxpool(x1)
        print(x2.size(),'x2')
        x3 = self.conv2(x2)
        print(x3.size(),'x3')
        x4= self.maxpool(x3)
        print(x4.size(),'x4')
        x5= self.conv3(x4)
        print(x5.size(),'x5')
        x6= self.maxpool(x5)
        print(x1.size(),'x1')
        x7 = self.conv4(x6)
        # print(x7.size())
        x8= self.maxpool(x7)
        # print(x8.size())
        x9 = self.conv5(x8)
        # print(x9.size())

        y9 = self.upsample1(x9)
        print(y9.size())
        y8 = self.deconv1(torch.cat((y9,crop_tensor(x7,y9)),axis=1))
        print(y8.size())
        y7 = self.upsample2(y8)
        print(y7.size())
        y6 = self.deconv2(torch.cat((y7,crop_tensor(x5,y7)),axis=1))
        print(y6.size())
        y5 = self.upsample3(y6)
        print(y5.size())
        y4 = self.deconv3(torch.cat((y5,crop_tensor(x3,y5)),axis=1))
        print(y4.size())
        y3 = self.upsample4(y4)
        print(y3.size())
        y2 = self.deconv4(torch.cat((y3,crop_tensor(x1,y3)),axis=1))
        y1 = self.out(y2)
        out = self.sigmoid(y1)

        return out



# X = torch.randn((1,3,572,572))
# model = UNet2()
# prediction= model(X)
# print(prediction.size())
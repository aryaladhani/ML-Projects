import torch 
import torch.nn as nn
# import torchvision

def convBlock(in_dim,out_dim):
    conv = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3, stride=1,padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )
    return conv

def deconvBlock(in_dim,out_dim):
    conv = nn.Sequential(
        # nn.ConvTranspose2d(in_dim,mid_dim, kernel_Size = 2, stride=2),
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=0),
        nn.BatchNorm2d(out_dim),
        nn.ReLU()
    )
    return conv

# def skip_conv(dim):
#     conv = nn.Sequential (
#         nn.Conv2d(dim,dim,kernel_size =5,padding=0),
#         nn.BatchNorm2d(dim),
#         nn.Conv2d(dim,dim,kernel_size =5),
#         nn.BatchNorm2d(dim)

#     )
#     return conv
def crop(tensor1,tensor2):
    tensor1_size = tensor1.size()[2]
    tensor2_size= tensor2.size()[2]
    delta = tensor1_size - tensor2_size
    delta = delta//2
    tensor1 = tensor1[:,:,delta:tensor1_size-delta,delta:tensor1_size-delta] 
    return tensor1
class UNet_skip_CNN(nn.Module):
    def __init__(self):
        super(UNet_skip_CNN,self).__init__()
        #enocder
        self.convBlock1= convBlock(3,64)
        self.MaxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.convBlock2= convBlock(64,128)
        self.convBlock3= convBlock(128,256)
        self.convBlock4= convBlock(256,512)
        self.convBlock5= convBlock(512,1024)
        self.skipblock2 = convBlock(128,128)
        self.skipblock3= convBlock(256,256)
        self.skipblock4 =  convBlock(512,512)
        self.skipblock1 = convBlock(64,64)
        #decoder 
        self.upSample1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.deconv1= deconvBlock(1024,512)
        self.upSample2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.deconv2= deconvBlock(512,256)
        self.upSample3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.deconv3= deconvBlock(256,128)
        self.upSample4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.deconv4= deconvBlock(128,64)
        self.out = nn.Conv2d(64,1,kernel_size=3,stride =1, padding =1)
        
    def forward(self,img):
        x1 = self.convBlock1(img)
        x2 = self.MaxPool(x1)
        x3 = self.convBlock2(x2)
        x4 = self.MaxPool(x3)
        x5 = self.convBlock3(x4)
        x6 = self.MaxPool(x5)
        x7 = self.convBlock4(x6)
        x8= self.MaxPool(x7)
        x9 = self.convBlock5(x8)
        
        y8= self.upSample1(x9)
        # return x7
        z7 = self.skipblock4(x7)
        y7= self.deconv1(torch.cat([y8,crop(z7,y8)],1))
        # return y7
        y6= self.upSample2(y7)
        z5 = self.skipblock3(x5)
        y5= self.deconv2(torch.cat([y6,crop(z5,y6)],1))
        y4= self.upSample3(y5)
        z3 = self.skipblock2(x3)
        y3= self.deconv3(torch.cat([y4,crop(z3,y4)],1))
        y2= self.upSample4(y3)
        z1 = self.skipblock1(x1)
        y1= self.deconv4(torch.cat([y2,crop(z1,y2)],1))
        out = self.out(y1)
        return out 

X = torch.randn((1,3,572,572))
model = UNet_skip_CNN()
prediction= model(X)
print(prediction.size())
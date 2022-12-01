import torch 
import torch.nn as nn
# import torchvision

def convBlock(in_dim,out_dim):
    conv1 = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3, stride=1,padding='same'),
        nn.ReLU()
    )
    conv2 = nn.Sequential(
        nn.Conv2d(out_dim,out_dim, kernel_size=3,stride =1 , padding = 'same'),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding='same'),
        nn.ReLU(),
        nn.Dropout(0.2)

    )
    
    return conv1, conv2 

def deconvBlock(in_dim,out_dim):
    conv1 = nn.Sequential(
        # nn.ConvTranspose2d(in_dim,mid_dim, kernel_Size = 2, stride=2),
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding='same'),
        nn.ReLU()
    )
    conv2 = nn.Sequential(
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding='same'),
        nn.ReLU(),
        nn.Dropout(0.2)
    )

    return conv1, conv2
def crop(tensor1,tensor2):
    tensor1_size = tensor1.size()[2]
    tensor2_size= tensor2.size()[2]
    delta = tensor1_size - tensor2_size
    delta = delta//2
    tensor1 = tensor1[:,:,delta:tensor1_size-delta,delta:tensor1_size-delta] 
    return tensor1
class UNet_ResNet(nn.Module):
    def __init__(self):
        super(UNet_ResNet,self).__init__()
        #enocder
        self.convBlock1a, self.convBlock1b = convBlock(3,64)
        self.MaxPool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.convBlock2a, self.convBlock2b = convBlock(64,128)
        self.convBlock3a, self.convBlock3b = convBlock(128,256)
        self.convBlock4a, self.convBlock4b = convBlock(256,512)

        #decoder 
        # self.upSample1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.deconv1a, self.deconv1b= deconvBlock(512,256)
        self.upSample2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.deconv2a, self.deconv2b= deconvBlock(256,128)
        self.upSample3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.deconv3a, self.deconv3b= deconvBlock(128,64)
        self.upSample4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.deconv4a, self.deconv4b= deconvBlock(128,64)
        self.out = nn.Conv2d(64,1,kernel_size=3,stride =1, padding =1)

    def forward(self,img):
        
        x1 = self.convBlock1a(img) #3-> #64
        x2 = self.convBlock1b(x1) #64->64
        x3 = self.MaxPool(x2) #64 ->64
        x4= self.convBlock2a(x3)#64 ->128
        x5 = self.convBlock2b(x4)
        x6= self.MaxPool(x5)
        x7 = self.convBlock3a(x6) #128->256
        x8 = self.convBlock3b(x7)
        # print(x8.size())
        x9 = self.MaxPool(x8)
        x10 = self.convBlock4a(x9) #256->512
        x11 = self.convBlock4b(x10)
        x12 = self.upSample2(x11) 
        # print(x12.size())
        x13 = self.deconv1a(torch.cat((x12,x8),axis =1)) #512->256
        x14 = self.deconv1b(x13)
        x15 = self.upSample3(x14)
        x16 = self.deconv2a(torch.cat((x15,x5),axis=1)) #256->128
        x17 = self.deconv2b(x16)
        x18 = self.upSample4(x17)
        x19 = self.deconv3a(torch.cat((x18,x2),axis=1))#128->64
        x20 = self.deconv3b(x19)
        out = self.out(x20)

        

        return out 


X = torch.randn((1,3,512,512))
model = UNet_ResNet()
prediction= model(X)
print(prediction.size())
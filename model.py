import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module): 
    def __init__(self):
        super(MyNet, self).__init__()
        
        ################################################################
        # TODO:                                                        #
        # Define your CNN model architecture. Note that the first      #
        # input channel is 3, and the output dimension is 10 (class).  #
        ################################################################
        
        self.seq1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size= (5,5),stride = 1),
            nn.Conv2d(in_channels=16,out_channels=10,kernel_size= (5,5),stride = 1),
            nn.BatchNorm2d(10,eps=0.0001,momentum=0.99),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=[5,5],stride=[1,1])
        )

        self.seq2=nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=64,kernel_size= (3,3),stride = 1),
            nn.BatchNorm2d(64,eps=0.001,momentum=0.99),       
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)

         )
   

        # self.seq3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64,out_channels=64,kernel_size= (1,10),stride = 1),
        #     nn.BatchNorm2d(64,eps=0.001,momentum=0.99),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=[2],stride=1)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(64*21*21, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 10)
        # )    
        self.fc=nn.Sequential(
            nn.Dropout1d(p=0.3),
            nn.Linear(in_features=64*21*21,out_features=32),
            nn.BatchNorm1d(32,eps=0.001,momentum=0.99),
            nn.ReLU(),
            nn.Dropout1d(p=0.3),
            nn.Linear(in_features=32,out_features=10),
         )
        pass

    def forward(self, x):
        
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################

        # x=self.se(x)
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        # x = nn.functional .softmax(x,-1)

        # pass
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        # model = ResNet18();
        
        super(ResNet18, self).__init__()
        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights)
        # print(self.resnet)
        # (batch_size, 512)

        self.resnet.conv1=nn. Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.resnet.maxpool=Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optinal):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################





    def forward(self, x):
        return self.resnet(x)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

        
    def forward(self, x):
        return x
    
if __name__ == '__main__':
    model = ResNet18()
    
    print(model)

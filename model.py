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

        # self.conv1 = nn.Conv2d(1, 6, (5, 5))   # output (N, C_{out}, H_{out}, W_{out})`
        # self.conv2 = nn.Conv2d(6, 16, (5, 5))
        # self.fc1 = nn.Linear(256, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # self.se=nn.Conv2d(in_channels=3,out_channels=3,kernel_size= (5,5),stride = 1)
        
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
        self.fc = nn.Sequential(
            nn.Linear(64*21*21, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10)
        )       

        # self.seq3 = nn.Sequential(
        #     nn.Conv2d(in_channels=64,out_channels=64,kernel_size= (1,10),stride = 1),
        #     nn.BatchNorm2d(64,eps=0.001,momentum=0.99),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=[1,5],stride=[1,2])
        # )


        # self.fc=nn.Sequential(
        #     nn.Dropout1d(p=0.5),
        #     nn.Linear(in_features=3*64*W*H,out_feature=32),
        #     nn.BatchNorm1d(32,eps=0.001,momentum=0.99),
        #     nn.ReLU(),
        #     nn.Dropout1d(p=0.3),
        #     nn.Linear(in_features=32,out_features=10),
        #     nn.Softmax()

        # )
        pass

    def forward(self, x):
        
        ##########################################
        # TODO:                                  #
        # Define the forward path of your model. #
        ##########################################
        # x=self.se(x)
        # print(x.shape)
        x = self.seq1(x)
        # print('seq1\n',x.shape)
        x = self.seq2(x)
        # print('seq2\n',x.shape)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        # print('fc\n',x.shape)
        x = nn.functional .softmax(x,-1)
        # print("sof.shape\n",x.shape)
        # pass
        return x
    
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        self.resnet = models.resnet18(pretrained=True)
        # (batch_size, 512)
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

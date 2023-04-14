import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

imput = torch.randn(1,3,128,128)
model= models.resnet18()
sub_modelback = nn.Sequential(*list(model.children())[1:-1])
class Mymodel(nn.Module):
  def _init_(self,output_class=5):
    super(Mymodel,self)._init_()
    self.subfirstlayerconv=nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    self.backbone = sub_modelback
    self.output = nn.Linear(512*4*4,output_class)
  def forward(self,x):
    x= self.backbone(x)
    batch_size = x.shape(0)
    x=x.reshape(batch_size,-1)
    x = self.output(x)
    return(x)  
  
updated_model = Mymodel()
intput_ = torch.randn(1,3,128,128)
output = updated_model(intput_)
print("output shape:{}".format(output.shape))
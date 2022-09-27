import torch
import torch.nn as nn
# from torch.autograd import Variable
# from dataset import CaptchaData
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose, ToTensor,ColorJitter,RandomRotation,RandomAffine,Resize,Normalize,CenterCrop,RandomApply,RandomErasing
import torchvision.models as models
# import time
# import copy

# import torchvision
# print(torchvision.__path__)
# print(torchvision.__version__)

import torch.onnx 

# 我看其他人都可以這樣創建...
model = models.densenet201(num_classes=1000, pretrained=True) # 0.15 pretrained 參數會移除
# model = models.densenet201(num_classes=1000, weights=DenseNet201_Weights.DEFAULT) # 未來 # 雖然我還不知道 DenseNet201_Weights 變數怎麼 import
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', pretrained=True)

# 設定
input_size = (1, 480, 360)

# set the model to inference mode 
model.eval() 

# Let's create a dummy input tensor  
# 這個是為了要讓 model 知道 input 大小是多少
dummy_input = torch.randn(1, 3, 480, 360, requires_grad=True)

# Export the model   
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "densenet201.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=10,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['modelInput'],   # the model's input names 
        output_names = ['modelOutput'], # the model's output names 
        dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                            'modelOutput' : {0 : 'batch_size'}}) 


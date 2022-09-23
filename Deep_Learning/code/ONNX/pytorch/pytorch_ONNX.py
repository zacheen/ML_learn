import os, sys
sys.path.append(os.getcwd())
import onnxruntime
import onnx
import cv2
import torch
import torchvision.models as models
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import tifffile

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([1024, 1024]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 讀取圖片
file_name = r"D:/our_data/bscan.JPEG"
# print(file_name)
# data_pic = tifffile.imread(file_name)
data_pic = Image.open(file_name)

# plt.imshow(data_pic)
# plt.show()

img = get_test_transform()(data_pic)
img = img.unsqueeze_(0) # -> NCHW, 1,3,224,224
print("input img mean {} and std {}".format(img.mean(), img.std()))

# read ONNX
onnx_model = onnxruntime.InferenceSession('../ONNX_model/DEJnet_densenet201.onnx')

inputs = {onnx_model.get_inputs()[0].name: to_numpy(img)}
outs = onnx_model.run(None, inputs)[0]

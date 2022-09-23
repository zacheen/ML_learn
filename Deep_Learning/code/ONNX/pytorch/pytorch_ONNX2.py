import torch
from onnx2torch import convert
import time
import onnx

# Path to ONNX model
onnx_model_path = r'E:\ONNX_model\DEJnet_densenet201.onnx'
# You can pass the path to the onnx model to convert it or...
torch_model_1 = convert(onnx_model_path)

# onnx_model = onnx.load(onnx_model_path)
# torch_model_2 = convert(onnx_model)
time.sleep(10);
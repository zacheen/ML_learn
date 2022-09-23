# wenet-main/wenet/bin/recognoize_onnx.py
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

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def get_test_transform():
    return transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 读取onnx，创建session
onnx_model = onnxruntime.InferenceSession('1.onnx')

inputs = {onnx_model.get_inputs()[0].name: to_numpy(img)}
outs = onnx_model.run(None, inputs)[0])

# 推理，获取输出，是一个列表
ort_outs = encoder_ort_session.run(None, ort_inputs)
encoder_out, encoder_out_lens, ctc_log_probs, beam_log_probs, beam_log_probs_idx = ort_outs
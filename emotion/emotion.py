'''
这个要写成打包好的,就是说,写成一个函数,直接要什么返回什么都在这里有
'''

import cv2
import torch
import torchvision.transforms as transforms
import argparse
import os
from .model import *

import sys

sys.path.append('../..')
from mtcnn import detect_faces
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def emotion(img, bbox, raise_hand):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load('/home/cpz/Desktop/AlphaPose/emotion/models/FER_trained_model.pt', map_location=lambda storage, loc: storage), strict=False)
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}
    val_transform = transforms.Compose([transforms.ToTensor()])

    xmin, xmax, ymin, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    # 这里的话不能用copy,因为一用copy的话,就不是同一个了,然后就画不回去,就不好
    # frame = img.copy()
    frame = img[ymin:ymax, xmin:xmax]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    det_frame = Image.fromarray(frame)
    bounding_boxes, landmarks = detect_faces(det_frame)

    if len(bounding_boxes) != 0:
        # print('bounding_boxes', bounding_boxes)
        fxmin, fymin, fxmax, fymax, confidence = bounding_boxes[0]
        fymin, fxmin, fymax, fxmax = max(int(fymin), 0), max(int(fxmin), 0), max(int(fymax), 0), max(int(fxmax), 0)

        cv2.rectangle(frame, (fxmin, fymin), (fxmax, fymax), (255, 0, 0), 2)
        resize_frame = cv2.resize(gray[fymin:fymax, fxmin:fxmax], (48, 48))
        X = resize_frame / 256
        X = Image.fromarray((X))
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            # log_ps = model.cpu()(X)
            log_ps = model(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())]
        if raise_hand == 1:
            cv2.putText(frame, 'raise hand', (fxmin, fymin-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        cv2.putText(frame, pred, (fxmin, fymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

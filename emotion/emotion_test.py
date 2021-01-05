import cv2
import torch
import torchvision.transforms as transforms
import argparse
import os
from model import *

import sys
sys.path.append('../..')
from mtcnn import detect_faces
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def load_trained_model(model_path):
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    return model


def FER_live_cam():
    model = load_trained_model('./models/FER_trained_model.pt')
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}
    val_transform = transforms.Compose([transforms.ToTensor()])
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det_frame = Image.fromarray(frame)
        bounding_boxes, landmarks = detect_faces(det_frame)

        for (xmin, ymin, xmax, ymax, confidence) in bounding_boxes:
            ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            resize_frame = cv2.resize(gray[ymin:ymax, xmin:xmax], (48, 48))
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
            cv2.putText(frame, pred, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    FER_live_cam()

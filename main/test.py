import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import sys
sys.path.append('../emotion')
from model import *
from mtcnn import detect_faces


def emotion():
    model = Face_Emotion_CNN()
    model.load_state_dict(torch.load('/home/cpz/Desktop/AlphaPose/emotion/models/FER_trained_model.pt', map_location=lambda storage, loc: storage), strict=False)
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}
    val_transform = transforms.Compose([transforms.ToTensor()])

    img = cv2.imread('../input/2.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    det_frame = Image.fromarray(img)
    bounding_boxes, landmarks = detect_faces(det_frame)

    for (xmin, ymin, xmax, ymax, confidence) in bounding_boxes:
        ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        resize_frame = cv2.resize(gray[ymin:ymax, xmin:xmax], (48, 48))

        X = resize_frame / 256
        X = Image.fromarray((resize_frame))
        X = val_transform(X).unsqueeze(0)
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = emotion_dict[int(top_class.numpy())]
        cv2.putText(img, pred, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.grid(False)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    emotion()

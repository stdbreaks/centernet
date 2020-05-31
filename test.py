import numpy as np
import os
import argparse
import cv2
import keras as keras
import tensorflow as tf
from keras.optimizers import Adam, Nadam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast,
RandomGamma, GaussianBlur, ToSepia, MotionBlur, InvertImg, HueSaturationValue, VerticalFlip, HorizontalFlip, ShiftScaleRotate, RandomShadow, RandomRain, Rotate)
from albumentations.core import composition
from albumentations.core.composition import BboxParams
from digi_gen import CSVGenerator
from loss import centernet_loss
from tqdm import tqdm
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter
import time

def top_k(array, k):
    tmp = array.copy()
    scores = np.full((k, ), -1.0, dtype=np.float32)
    x = np.full((k, ), -1.0, dtype=np.float32)
    y = np.full((k, ), -1.0, dtype=np.float32)
    classes = np.full((k, ), -1.0, dtype=np.float32)

    tmp = tmp.flatten()

    for i in range(k):
        idx = tmp.argmax()
        scores[i] = tmp[idx]
        y[i], x[i], classes[i] = np.unravel_index(idx, array.shape)
        tmp[idx] = -1.0
    
    top_array = np.stack([scores, x, y, classes], axis=-1)
    return top_array

model = load_model('./models/centernet_val_25.h5', compile=False)
model.compile(optimizer=Nadam(), loss=centernet_loss(10))

imgs = list(os.listdir('/storage/stdbreaks/bread_conveyor/cut/'))
avg_time = 0
for img_name in tqdm(imgs):
    
    img_path = os.path.join('/storage/stdbreaks/bread_conveyor/cut/', img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    orig_img = img.copy()

    
    img = cv2.resize(img, dsize=(320, 320), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32)
    img /= 255.
    img = np.reshape(img, (1, img.shape[0], img.shape[1], 3))

    output = model.predict(img)
    scales = output[0, :, :, 1:]
    output = output[0, :, :, :1]

    out_pool = np.zeros_like(output)
    for c in range(output.shape[2]):
        out_pool[:,:,c] = maximum_filter(output[:,:,c], size=5, mode='constant', cval=0.)
    kekw = np.where(out_pool == output, output, 0)
    kekw[kekw < 0.4] = 0
    centers = top_k(kekw, 60)
    filtered = []
    for center in centers:
        if center[0] > 0:
            filtered.append(center.tolist())

    filtered.sort(key=lambda x: x[1])

    test_filt = []
    for filt in filtered:
        test_filt.append(filt.copy())

    for a in range(len(filtered)):
        for b in range(len(filtered)):
            dx = abs(filtered[a][1] - filtered[b][1])
            dy = abs(filtered[a][2] - filtered[b][2])
            if dx < 10 and dy < 10 and filtered[a] != filtered[b]:
                if filtered[a][0] > filtered[b][0]:
                    test_filt[b] = []
                else:
                    test_filt[a] = []
        

    out_number = ''
    img_signed = orig_img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255)]
    for center in test_filt:
        if len(center) != 0:
            bboxh = scales[int(center[2]), int(center[1]), 0]
            bboxw = scales[int(center[2]), int(center[1]), 1]
            bboxh2 = bboxh * orig_img.shape[0] / 2
            bboxw2 = bboxw * orig_img.shape[1] / 2

            center_x = center[1] * (orig_img.shape[1] / scales.shape[1])
            center_y = center[2] * (orig_img.shape[0] / scales.shape[0])

            xmin = int(center_x - bboxw2)
            ymin = int(center_y - bboxh2)
            xmax = int(center_x + bboxw2)
            ymax = int(center_y + bboxh2)

            cv2.rectangle(img_signed, (xmin, ymin), (xmax, ymax), colors[int(center[-1])], 2, cv2.LINE_AA)

            out_number += str(int(center[-1]))

    cv2.imwrite(os.path.join('/storage/stdbreaks/bread_conveyor/centernet/test/',img_name), img_signed)

avg_fps = avg_time / len(imgs)
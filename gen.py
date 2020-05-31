import os
import csv
import numpy as np
import cv2
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast,
RandomGamma, GaussianBlur, MotionBlur, ToSepia, InvertImg, RandomSnow, RandomSunFlare, RandomRain, RandomShadow, HueSaturationValue, HorizontalFlip)
from albumentations import BboxParams
from keras.utils import Sequence
from matplotlib import pyplot as plt

def strong_aug(p=0.6):
    return Compose([
        RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.75),
        OneOf([
            MotionBlur(),
            GaussianBlur()
        ]),
        OneOf([
            ToGray(),
            ToSepia()
        ]),
        OneOf([
            InvertImg(),
            RandomBrightnessContrast(brightness_limit=0.75, p=0.75),
            RandomGamma(),
            HueSaturationValue()
        ], p=0.75)
    ], bbox_params=BboxParams("pascal_voc", label_fields=["category_id"], min_area=0.0, min_visibility=0.0), p=p)


class CSVGenerator(Sequence):
    def __init__(self, annotations_path, 
                 classes_path, 
                 img_height, 
                 img_width, 
                 batch_size,
                 augs):
        self.classes_path = classes_path
        self.annotations_path = annotations_path
        self.annotations = {}
        self.imgs_list = []
        self.out_height = img_height
        self.out_width = img_width
        self.augs = augs
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        self.classes = {}
        self.num_classes = 0
        with open(self.classes_path, 'r') as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                self.classes[row[0]] = int(row[1])
                self.num_classes += 1

        with open(self.annotations_path, 'r') as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                if row[0] not in self.annotations:
                    self.annotations[row[0]] = []
                
                xmin = -1
                ymin = -1
                xmax = -1
                ymax = -1

                if row[1] == '':
                    xmin = -1
                else:
                    xmin = int(row[1])
                
                if row[2] == '':
                    ymin = -1
                else:
                    ymin = int(row[2])

                if row[3] == '':
                    xmax = -1
                else:
                    xmax = int(row[3])
                
                if row[4] == '':
                    ymax = -1
                else:
                    ymax = int(row[4])

                class_id = 0

                annotation = [xmin, ymin, xmax, ymax, class_id]
                self.annotations[row[0]].append(annotation)

        self.imgs_list = list(self.annotations.keys())

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.imgs_list) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.imgs_list[idx * self.batch_size : (idx + 1) * self.batch_size]

        imgs = np.empty((self.batch_size, self.img_height, self.img_width, 3), dtype=np.float32)
        batch_centers = np.zeros((self.batch_size, self.out_height, self.out_width, self.num_classes + 2), dtype=np.float32)
        
        for i in range(len(batch)):
            img = cv2.imread(batch[i], cv2.IMREAD_COLOR)
            if img is None:
                print(batch[i])
            orig_h = img.shape[0]
            orig_w = img.shape[1]
            
            annotations = []
            for ann in self.annotations[batch[i]]:
                annotations.append(ann.copy())

            img = cv2.resize(img, dsize=(self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
            img_h = img.shape[0]
            img_w = img.shape[1]
            img_scale_h = orig_h / img.shape[0]
            img_scale_w = orig_w / img.shape[1]

            for j in range(len(annotations)):
                annotations[j][0] = int(np.round(annotations[j][0] / img_scale_w))
                annotations[j][1] = int(np.round(annotations[j][1] / img_scale_h))
                annotations[j][2] = int(np.round(annotations[j][2] / img_scale_w))
                annotations[j][3] = int(np.round(annotations[j][3] / img_scale_h))
            

            if self.augs is not None:
                boxes = []
                category_ids = []

                for j in range(len(annotations)):
                    boxes.append([annotations[j][0], annotations[j][1],
                                annotations[j][2], annotations[j][3]])
                    category_ids.append(annotations[j][4])

            
                data = {'image': img, 'bboxes': boxes, 'category_id':category_ids}
                augmented = self.augs(**data)
                img = augmented['image']

            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #img = self.clahe.apply(img)
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            #img = np.reshape(img, (img.shape[0], img.shape[1], 1))

            img = img.astype(np.float32)
            img /= 255.
            imgs[i] = img

            if self.augs is not None:
                annotations = []

                for j in range(len(augmented['bboxes'])):
                    annotations.append([int(np.round(augmented['bboxes'][j][0])),
                                        int(np.round(augmented['bboxes'][j][1])),
                                        int(np.round(augmented['bboxes'][j][2])),
                                        int(np.round(augmented['bboxes'][j][3])),
                                        augmented['category_id'][j]])


            centers = np.zeros((self.out_height, self.out_width, self.num_classes), dtype=np.float32)
            scales = np.zeros((self.out_height, self.out_width, 2), dtype=np.float32)

            for j, bbox in enumerate(annotations):
                #orig_bbox = bbox.copy()
                #cv2.rectangle(img, (orig_bbox[0], orig_bbox[1]), (orig_bbox[2], orig_bbox[3]), (0, 255, 0), 2)
                #bbox = [int(np.round(bbox[0])), int(np.round(bbox[1])), int(np.round(bbox[2])), int(np.round(bbox[3])), int(bbox[4])]
                if (bbox[0] == 0) or (bbox[1] == 0) or (bbox[2] == 0) or (bbox[3] == 0) or (bbox[2] - bbox[0] == 0) or (bbox[3] - bbox[1] == 0):
                    continue
                center_x = int((bbox[2] + bbox[0]) / 2)
                if center_x == self.out_width:
                    center_x -= 1

                center_y = int((bbox[3] + bbox[1]) / 2)
                
                if center_y >= self.out_height:
                    center_y = self.out_height - 1

                h = bbox[3] - bbox[1]
                sc_h = h / img_h
                if sc_h > 1.0:
                    sc_h = 1.0
                w = bbox[2] - bbox[0]
                sc_w = w / img_w
                if sc_w > 1.0:
                    sc_w = 1.0

                xmin = int(bbox[0]-1)
                xmax = int(bbox[2]-1)
                ymin = int(bbox[1]-1)
                ymax = int(bbox[3]-1)

                if ymax >= centers.shape[0]:
                    ymax = centers.shape[0] - 1

                if xmax >= centers.shape[1]:
                    xmax = centers.shape[1] - 1

                rhmin = -int(h/2)
                rhmax = int(h/2)
                rwmin = -int(w/2)
                rwmax = int(w/2)

                if h % 2 != 0:
                    rhmax = int(h/2) + 1

                if w % 2 != 0:
                    rwmax = int(w/2) + 1 

                y, x = np.ogrid[rhmin:rhmax,rwmin:rwmax]

                e = np.exp(-((x*x/(2*(w/12.0)*(w/12.0)) + (y*y/(2*(h/12.0)*(h/12.0))))))

                tmp = np.zeros_like(centers)
                tmp[ymin:ymax, xmin:xmax, 0] = e

                centers = np.where(tmp > centers, tmp, centers)

                #for x in range(xmin, xmax + 1):
                #    for y in range(ymin, ymax + 1):
                #        conf = np.exp((-1.0) * (
                #            (np.square(x - center_x)/(2.0 * (w / 3.))) +
                #            (np.square(y - center_y)/(2.0 * (h / 3.)))
                #        ))

                #        if conf > centers[y, x, int(bbox[4])]:
                #            centers[y, x, int(bbox[4])] = conf
                centers[int(center_y-1), int(center_x-1), 0] = 1.0
                scales[int(center_y-1), int(center_x-1), 0] = sc_h
                scales[int(center_y-1), int(center_x-1), 1] = sc_w

            #cv2.imshow('tester', tester1)
            #cv2.waitKey()
            batch_centers[i] = np.concatenate([centers, scales], axis=2)

            #print(labels.shape)

            return imgs, batch_centers

if __name__ == '__main__':
    gen = CSVGenerator('train.csv', 'class.csv', 640, 1024, 1, None)
    for i in range(2583):
        imgs, centers = gen.__getitem__(i)

        plt.imshow(centers[0, :, :, 3])
        plt.show()
        print(np.sum(centers[0, :, :, 3]))
        print(np.count_nonzero(centers[0, :, :, 3]))

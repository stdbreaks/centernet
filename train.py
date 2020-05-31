import numpy as np
import os
import argparse
import cv2
import keras as keras
import tensorflow as tf
from keras.optimizers import Adam, Nadam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from albumentations import (ToGray, OneOf, Compose, RandomBrightnessContrast, JpegCompression, Cutout, GaussNoise, IAAAffine, IAAPerspective,
RandomGamma, GaussianBlur, ToSepia, MotionBlur, InvertImg, Transpose, HueSaturationValue, VerticalFlip, HorizontalFlip, ShiftScaleRotate, RandomShadow, RandomRain, Rotate)
from albumentations.core import composition
from albumentations.core.composition import BboxParams
from gen import CSVGenerator
from loss import centernet_loss, center_neg_loss, center_pos_loss, reg_loss
from segmentation_models.losses import bce_dice_loss, dice_loss
import segmentation_models as sm

def strong_aug(p=0.75):
    return Compose([
        ShiftScaleRotate(scale_limit=0.1, rotate_limit=90),
        Transpose(),
        #IAAAffine(shear=0.1),
        #IAAPerspective(),
        Cutout(num_holes=20, max_h_size=8, max_w_size=8),
        HorizontalFlip(),
        VerticalFlip(),
        GaussNoise(),
        JpegCompression(),
        #RandomShadow(shadow_roi=(0, 0, 1, 1), p=0.75),
        OneOf([
            MotionBlur(),
            GaussianBlur()
        ]),
        OneOf([
            ToGray(),
            ToSepia()
        ]),
        RandomBrightnessContrast(brightness_limit=0.75, p=0.75)
    ], bbox_params=BboxParams("pascal_voc", label_fields=["category_id"], min_area=0.0, min_visibility=0.5), p=p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_annotations', type=str)
    parser.add_argument('val_annotations', type=str)
    parser.add_argument('classes', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('--image-height', type=int)
    parser.add_argument('--image-width', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--alpha-neg', type=float, default=3.0)
    parser.add_argument('--alpha-pos', type=float, default=3.0)
    parser.add_argument('--near-center-coef', type=float, default=4.0)
    parser.add_argument('--negatives-weight', type=float, default=0.25)
    parser.add_argument('--regression-weight', type=float, default=10.0)
    args = parser.parse_args()

    train_generator = CSVGenerator(args.train_annotations, args.classes,
                                   args.image_height, args.image_width, 
                                   args.batch_size, strong_aug())
    val_generator = CSVGenerator(args.val_annotations, args.classes,
                                 args.image_height, args.image_width,
                                 1, None)

    callbacks = [
        ModelCheckpoint(os.path.join(args.model_path,'centernet_val_{epoch}.h5'), 
                        monitor='val_loss', 
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=False, 
                        mode='min'),
        ModelCheckpoint(os.path.join(args.model_path, 'centernet_loss_{epoch}.h5'), 
                        monitor='loss', 
                        verbose=1, 
                        save_best_only=True, 
                        save_weights_only=False, 
                        mode='min'),
        ReduceLROnPlateau(monitor='val_loss', 
                          factor=0.25, 
                          patience=3, 
                          verbose=1, 
                          mode='min')
    ]

    model = sm.Linknet('resnet18', input_shape=(args.image_height, args.image_width, 3), classes=train_generator.classes+2, activation='sigmoid')
    model.summary()

    loss = centernet_loss(train_generator.num_classes, args.alpha_pos, args.alpha_neg, args.near_center_coef, args.negatives_weight, args.regression_weight)

    opt = Nadam(lr=args.lr)

    model.compile(opt, loss=loss, metrics=[center_pos_loss(train_generator.num_classes), center_neg_loss(train_generator.num_classes), reg_loss(train_generator.num_classes)])

    model.fit_generator(train_generator, steps_per_epoch=args.steps, validation_data=val_generator,
                        epochs=args.epochs, verbose=1, callbacks=callbacks, max_queue_size=64)
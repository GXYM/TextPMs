import subprocess
import os
import numpy as np
import cv2
import torch
import time
from scipy  import stats
from util.config import config as cfg
from scipy import ndimage as ndimg
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def sigmoid_alpha(x, k):
    betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
    dm = max(np.max(x), 0.0001)
    res = (2 / (1 + np.exp(-x*k/dm)) - 1)*betak
    return np.maximum(0, res)


def Evalue(mask, beta):
    alpha = cfg.fuc_k
    Ev = list()
    pt_sum = np.sum(mask)
    dmp = ndimg.distance_transform_edt(mask)  # distance transform
    for k in alpha[beta:]:
        alpha_mask = sigmoid_alpha(dmp, k)
        e = (np.sum(alpha_mask)/pt_sum)
        Ev.append(e)

    return np.array(Ev), np.max(dmp)


def pse_warpper(kernals, min_area=10):
    '''
    reference https://github.com/liuheng92/tensorflow_PSENet/blob/feature_dev/pse
    :param kernals:
    :param min_area:
    :return:
    '''
    from .pse import pse_cpp
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    label_num, label = cv2.connectedComponents(kernals[0].astype(np.uint8), connectivity=8)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)

    pred = pse_cpp(label, kernals, c=kernal_num)

    return np.array(pred), label_values


def decode(preds, scale, threshold=0.33, voting=False):
    """
    generating and filtering the text instance from pms
    """
    t0 = time.time()
    post_time = 0
    #preds = torch.sigmoid(preds)
    #preds = preds.detach().cpu().numpy()

    score_0 = preds[-1].astype(np.float32)
    score = preds[0:].astype(np.float32)
    preds = preds > threshold
    pred, label_values = pse_warpper(preds)
    bbox_list = []
    polygons = []
    for label_value in label_values:
        mask = pred == label_value
        if np.sum(mask) < cfg.min_area /(scale * scale):  # 150 / 300
            continue
        if voting:
            e_value, text_scale = Evalue(np.array(mask, dtype=np.uint8), 0)
            score_i = np.mean(score[:, pred == label_value], axis=1)
            #print((e_value, score_i))
            weight = np.array([0.1,0.2,0.3,0.4])
            #Rvote = np.array((score_i >= e_value-threshold/2)* (score_i <=e_value+threshold/2),  dtype=np.uint8)
            #ResFlag = np.sum(np.array(score_i >= e_value-threshold/2,  dtype=np.uint8))
            Rvote = np.array(score_i >= e_value-threshold**2,  dtype=np.uint8)
            Rscore = weight*Rvote
            #print(score_i -e_value)
            if np.sum(Rscore) < 0.5:
                continue
        else:
            score_i0 = np.mean(score_0[pred == label_value])
            if score_i0 < cfg.score_i:
                continue

        # binary，contours，hierarchy
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        post_time = time.time() - t0

        rect = cv2.minAreaRect(contours[0])
        points = cv2.boxPoints(rect)
        points = np.int0(points)
        # 轮廓近似，epsilon数值越小，越近似
        epsilon = 0.007 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        polygons.append(approx.reshape((-1, 2))*[scale, scale])

        bbox_list.append(points*[scale,scale])

    return pred, bbox_list, polygons, post_time

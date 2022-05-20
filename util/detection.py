import time
import cv2
import torch
import numpy as np
from util.config import config as cfg
from util.misc import fill_hole
from skimage import segmentation


def sigmoid_alpha(x, k):
    betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
    dm = max(np.max(x), 0.0001)
    res = (2 / (1 + np.exp(-x * k / dm)) - 1) * betak
    return np.maximum(0, res)


def watershed_segment(preds, scale=1.0):
    text_region = np.mean(preds[2:], axis=0)
    region = fill_hole(text_region >= cfg.threshold)

    text_kernal = np.mean(preds[0:2], axis=0)
    kernal = fill_hole(text_kernal >= cfg.threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(kernal, cv2.MORPH_OPEN, kernel, iterations=1)
    kernal = cv2.erode(opening, kernel, iterations=1)  # sure foreground area
    ret, m = cv2.connectedComponents(kernal)

    distance = np.mean(preds[:], axis=0)
    distance = np.array(distance / np.max(distance) * 255, dtype=np.uint8)
    labels = segmentation.watershed(-distance, m, mask=region)
    boxes = []
    contours = []
    small_area = (300 if cfg.test_size[0] >= 256 else 150)
    for idx in range(1, np.max(labels) + 1):
        text_mask = labels == idx
        if np.sum(text_mask) < small_area / (cfg.scale * cfg.scale) \
                or np.mean(preds[-1][text_mask]) < cfg.score_i:  # 150 / 300
            continue
        cont, _ = cv2.findContours(text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        epsilon = 0.003 * cv2.arcLength(cont[0], True)
        approx = cv2.approxPolyDP(cont[0], epsilon, True)
        contours.append(approx.reshape((-1, 2)) * [scale, scale])

    return labels, boxes, contours


class TextDetector(object):

    def __init__(self, model):
        # evaluation mode
        self.model = model
        self.model.eval()

    def detect(self, image, img_show=None):
        # get model output
        with torch.no_grad():
            b, c, h, w = image.shape
            img = torch.ones((b, c, cfg.test_size[1], cfg.test_size[1]), dtype=torch.float32).cuda()
            img[:,:, :h, :w] = image[:, :, :, :]
            if cfg.exp_name != "Icdar2015":
                preds, backbone_time, iter_time = self.model.forward(img)
            else:
                preds, backbone_time, iter_time = self.model.forward(image)
            preds = torch.sigmoid(preds[0, :, :h//cfg.scale, :w//cfg.scale])

        t0 = time.time()
        preds = preds.detach().cpu().numpy()
        detach_time = time.time() - t0

        if cfg.recover == "watershed":
            t0 = time.time()
            labels, boxes, contours = watershed_segment(preds, scale=cfg.scale)
            post_time = time.time() - t0
        elif cfg.recover == "pse":
            from pse import decode as pse_decode
            t0 = time.time()
            labels, boxes, contours = pse_decode(preds, cfg.scale, cfg.threshold)
            post_time = time.time() - t0
        else:
            print("Not Supported !")

        output = {
            'image': image,
            'tr': labels,
            'bbox': boxes,
            "backbone_time": backbone_time,
            "iter_time": iter_time,
            "detach_time": detach_time,
            "post_time": post_time
        }
        return contours, output























import numpy as np
import cv2
# c++ version pse based on opencv 3+
from postprogress import decode as pse_decode
from util.config import config as cfg
import torch
import time

class TextDetector(object):

    def __init__(self, model):
        # evaluation mode
        self.model = model
        model.eval()
        # parameter
        self.scale = cfg.scale
        self.threshold = cfg.threshold
        self.IN_SIZE = cfg.test_size


    def detect(self, image, img_show):
        # get model output
        b, c, h, w = image.shape
        img = torch.ones((b, c, self.IN_SIZE[1], self.IN_SIZE[1]), dtype=torch.float32).cuda()
        img[:,:, :h, :w]= image[:, :, :, :]
        tt = time.time()
        if cfg.exp_name != "Icdar2015":
            preds, backbone_time, IM_time = self.model.forward(img)
        else:
            preds, backbone_time, IM_time = self.model.forward(image)
        preds = torch.sigmoid(preds[0, :, :h//self.scale, :w//self.scale])
        #net_time = time.time() - tt
        t0 = time.time()
        preds = preds.detach().cpu().numpy()
        detach_time = time.time() - t0
        net_time = time.time() - tt
        preds, boxes, contours, post_time = pse_decode(preds, self.scale, self.threshold, voting=cfg.voting)
        output = {
            'image': image,
            'tr': preds,
            'bbox': boxes,
             "backbone_time": backbone_time,
             "IM_time": IM_time,
             "detach_time": detach_time
        }
        return contours, output, net_time, post_time























import torch
import numpy as np
import cv2
import os
from util.config import config as cfg
from util import canvas as cav


def visualize_network_output(output, tr_mask, mode='train'):

    vis_dir = os.path.join(cfg.vis_dir, cfg.exp_name + '_' + mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)
    b, c, _, _ = output.shape

    for i in range(b):
        predict = torch.sigmoid(output[i]).data.cpu().numpy()
        target = tr_mask[i].cpu().numpy()
        shows = list()
        for j in range(c):
            p = predict[j]
            t = target[:, :, j]
            tcl_pred = cav.heatmap(np.array(p / np.max(p) * 255, dtype=np.uint8))
            tcl_targ = cav.heatmap(np.array(t / np.max(t) * 255, dtype=np.uint8))
            show = np.concatenate([tcl_pred * 255, tcl_targ * 255], axis=0)
            shows.append(show)
        show_img = np.concatenate(shows, axis=1)

        show = cv2.resize(show_img, (256*c, 512))
        path = os.path.join(vis_dir, '{}.png'.format(i))
        cv2.imwrite(path, show)


def visualize_gt(image, contours, tr=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    image_show = cv2.polylines(image_show, contours, True, (0, 0, 255), 3)
    if tr is not None:
        tr_map = cav.heatmap(np.array(tr[:, :, -1] / np.max(tr[:, :, -1]) * 255, dtype=np.uint8))
        h, w = tr_map.shape[:2]
        tr_map = cv2.resize(tr_map, (w*cfg.scale, h*cfg.scale))
        image_show = np.concatenate([image_show, np.array(tr_map*255, dtype=np.uint8)], axis=1)
        return image_show
    else:
        return image_show


def visualize_detection(image, contours, tr=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    cv2.drawContours(image_show, contours, -1, (0, 255, 0), 3)
    if tr is not None:
        tr_map = cav.heatmap(np.array(tr / np.max(tr) * 255, dtype=np.uint8))
        h,w = tr_map.shape[:2]
        tr_map = cv2.resize(tr_map, (w*cfg.scale, h*cfg.scale))
        #print(tr_map.shape)
        #print(image_show.shape)
        image_show = np.concatenate([image_show, np.array(tr_map*255, dtype=np.uint8)], axis=1)
        return image_show
    else:
        return image_show

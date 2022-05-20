import copy
import cv2
import torch
import numpy as np
from PIL import Image
from scipy import ndimage as ndimg
from util.config import config as cfg
from util.misc import find_bottom, find_long_edges, split_edge_seqence, norm2, split_edge_seqence_by_step
import math


def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):
    def __init__(self, points, orient, text):
        self.orient = orient
        self.text = text
        self.bottoms = None
        self.e1 = None
        self.e2 = None
        if self.text != "#":
            self.label = 1
        else:
            self.label = -1

        remove_points = []
        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area)/ori_area < 0.0017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=15):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def Equal_width_bbox_cover(self, step=16.0):

        inner_points1, inner_points2 = split_edge_seqence_by_step(self.points, self.e1, self.e2, step=step)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(object):

    def __init__(self, transform, is_training=False):
        super().__init__()
        self.transform = transform
        self.is_training = is_training
        self.scale = cfg.scale
        self.alpha = cfg.fuc_k
        self.mask_cnt = len(cfg.fuc_k)

    def sigmoid_alpha(self, x, k):
        betak = (1 + np.exp(-k)) / (1 - np.exp(-k))
        dm = max(np.max(x), 0.0001)
        res = (2 / (1 + np.exp(-x*k/dm)) - 1)*betak
        return np.maximum(0, res)

    def make_text_region(self, img, polygons):
        h, w = img.shape[0]//self.scale, img.shape[1]//self.scale
        mask_ones = np.ones(img.shape[:2], np.uint8)
        mask_zeros = np.zeros(img.shape[:2], np.uint8)

        train_mask = np.ones((h, w), np.uint8)
        tr_mask = np.zeros((h, w, self.mask_cnt), np.float)
        if polygons is None:
            return tr_mask, train_mask

        for polygon in polygons:
            instance_mask = mask_zeros.copy()
            cv2.fillPoly(instance_mask, [polygon.points.astype(np.int32)], color=(1,))
            dmp = ndimg.distance_transform_edt(instance_mask[::self.scale, ::self.scale])  # distance transform
            for i, k in enumerate(self.alpha):
                tr_mask[:, :, i] = np.maximum(tr_mask[:, :, i], self.sigmoid_alpha(dmp, k))

            if polygon.text == '#':
                cv2.fillPoly(mask_ones, [polygon.points.astype(np.int32)], color=(0,))
                continue

        train_mask = mask_ones[::self.scale, ::self.scale]

        return tr_mask, train_mask

    def get_training_data(self, image, polygons, image_id, image_path):

        H, W, _ = image.shape
        if self.transform:
            image, polygons = self.transform(image, copy.copy(polygons))
            h, w, _ = image.shape

        tr_mask, train_mask = self.make_text_region(image, polygons)
        # clip value (0, 1)
        tr_mask = np.clip(tr_mask, 0, 1)
        train_mask = np.clip(train_mask, 0, 1)

        # # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        if not self.is_training:
            points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
            length = np.zeros(cfg.max_annotation, dtype=int)
            if polygons is not None:
                for i, polygon in enumerate(polygons):
                    pts = polygon.points
                    points[i, :pts.shape[0]] = polygon.points
                    length[i] = pts.shape[0]

            meta = {
                'image_id': image_id,
                'image_path': image_path,
                'annotation': points,
                'n_annotation': length,
                'Height': H,
                'Width': W
            }

            return image, train_mask, tr_mask, meta

        image = torch.from_numpy(image).float()
        train_mask = torch.from_numpy(train_mask).byte()
        tr_mask = torch.from_numpy(tr_mask).float()

        return image, train_mask, tr_mask

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()

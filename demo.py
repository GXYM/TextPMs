import os
import time
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset import DeployDataset
from network.textnet import TextNet
from util.config import config as cfg, update_config, print_config
from util.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_detection, visualize_gt
from util.detection import TextDetector
from util.misc import to_device, mkdirs,rescale_result


import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)


def write_to_file(contours, file_path):
    """
    :param contours: [[x1, y1], [x2, y2]... [xn, yn]]
    :param file_path: target file path
    """
    # according to total-text evaluation method, output file shoud be formatted to: y0,x0, ..... yn,xn
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')


def inference(detector, test_loader, output_dir):

    total_time = 0.
    if cfg.exp_name != "MLT2017" and cfg.exp_name != "ArT":
        osmkdir(output_dir)
    else:
        if not os.path.exists(output_dir):
            mkdirs(output_dir)
        if cfg.exp_name == "MLT2017":
            out_dir = os.path.join(output_dir, "{}_{}_{}_{}_{}".
                                   format(str(cfg.checkepoch), cfg.test_size[0],
                                          cfg.test_size[1], cfg.dis_threshold, cfg.cls_threshold))
            if not os.path.exists(out_dir):
                mkdirs(out_dir)

    for i, (image, meta) in enumerate(test_loader):

        idx = 0  # test mode can only run with batch_size == 1
        image = to_device(image)

        start = time.time()
        torch.cuda.synchronize()
        contours, output = detector.detect(image)
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0

        print('detect {} / {} images: {}. ({:.2f} fps); '
              .format(i + 1, len(test_loader), meta['image_id'][idx], fps))

        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if cfg.viz:
            pred_vis = visualize_detection(img_show, contours, output['tr'])
            path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                                meta['image_id'][idx].split(".")[0] + ".jpg")
            cv2.imwrite(path, pred_vis)

        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
        img_show, contours = rescale_result(img_show, contours, H, W)
        fname = meta['image_id'][idx].replace('jpg', 'txt')
        write_to_file(contours, os.path.join(output_dir, fname))


def main(vis_dir_path):

    osmkdir(vis_dir_path)
    testset = DeployDataset(
        image_root=cfg.img_root,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    )

    if cfg.cuda:
        cudnn.benchmark = True

    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'TextPMs_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))

    # copy to cuda
    model.load_model(model_path)
    model = model.to(cfg.device)
    detector = TextDetector(model)

    print('Start testing TextPMs.')
    output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    inference(detector, test_loader, output_dir)


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))

    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)

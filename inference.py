import os

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from tqdm import tqdm
from argparse import Namespace
from typing import List
from PIL import Image

# from Codes.src.lightning_scripts import main
from Codes.src.lightning_scripts.trainers import thermal_edge_trainer
from Codes.src.core.utils.filesystem import checkpoint
from Codes.src.core.utils import collect_env_info, get_rank, setup_logger
from Codes.src.core.callbacks import ProgressBar


def main(args):
    ckp = checkpoint(args)
    # seeding
    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)
    log_dir = ckp.get_path("logs")
    logger = setup_logger(name="pytorch_lightning",
                          save_dir=log_dir,
                          distributed_rank=get_rank(),
                          color=True,
                          abbrev_name=None,
                          print_to_console=True if args.debug else False)
    logger.info("Environment info:\n" + collect_env_info())

    if args.test_checkpoint is not None:
        t_checkpoint = args.test_checkpoint
    elif args.test_monitor is not None and os.path.exists(args.test_monitor_path):
        best = 0.0
        logger.info("Searching in {}".format(args.test_monitor_path))
        for x in os.listdir(args.test_monitor_path):
            if args.test_monitor in x:
                val = float(x[-11:-5])
                if val >= best:
                    t_checkpoint = os.path.join(args.test_monitor_path, x)
                    logger.info("Found {}".format(t_checkpoint))
                    best = val
        logger.info("Final best checkpoint is {}".format(t_checkpoint))
    else:
        ValueError("Provide the checkpoint for testing")

    logger.info("Loading from {}".format(t_checkpoint))

    model = thermal_edge_trainer.load_from_checkpoint(checkpoint_path=t_checkpoint,
                                                      args=args,
                                                      ckp=ckp,
                                                      train=False,
                                                      logger=logger,
                                                      )
    trainer = pl.Trainer(gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         max_epochs=1,
                         distributed_backend="dp",
                         amp_level="O0",
                         callbacks=[ProgressBar(logger)],
                         fast_dev_run=False,
                         progress_bar_refresh_rate=0,
                         deterministic=True,
                         replace_sampler_ddp=False,
                         )
    trainer.test(model=model)


def seg2edge(seg: np.ndarray, radius: int, edge_type: str):
    """
    This function takes an input segment and produces binary bdrys - boundaries.
    Multichannel input segments are supported by the function.
    Code Adapted from
    https://github.com/Lavender105/DFF/blob/152397cec4a3dac2aa86e92a65cc27e6c8016ab9/lib/matlab/modules/data/seg2edge.m
    Args:
        seg: image of segmentation.
        radius: radius of neighborhood.
        edge_type: what edge type to generate.
    """
    # Get dimensions
    height, width, chn = seg.shape

    # Set the considered neighborhood
    radius_search = max(int(np.ceil(radius)), 1)
    X, Y = np.meshgrid(np.arange(1, width + 1), np.arange(1, height + 1))
    x, y = np.meshgrid(np.arange(-radius_search, radius_search + 1), np.arange(-radius_search, radius_search + 1))

    # Columnize everything
    X = X.ravel()
    Y = Y.ravel()
    x = x.ravel()
    y = y.ravel()
    if chn == 1:
        seg = seg.ravel()
    else:
        seg = seg.reshape(height * width, chn)

    # Build circular neighborhood
    idxNeigh = np.sqrt(x ** 2 + y ** 2) <= radius
    x = x[idxNeigh]
    y = y[idxNeigh]
    numPxlImg = len(X)
    numPxlNeigh = len(x)

    # Compute Gaussian weight
    idxEdge = np.zeros(numPxlImg, dtype=bool)
    for i in range(numPxlNeigh):
        XNeigh = X + x[i]
        YNeigh = Y + y[i]
        idxValid = np.where(
            (XNeigh >= 1) & (XNeigh <= width) & (YNeigh >= 1) & (YNeigh <= height)
        )[0]  # filtering out of boundaries "neighbours"

        XCenter = X[idxValid]
        YCenter = Y[idxValid]
        XNeigh = XNeigh[idxValid]
        YNeigh = YNeigh[idxValid]
        LCenter = seg[np.ravel_multi_index((YCenter - 1, XCenter - 1), (height, width))]
        LNeigh = seg[np.ravel_multi_index((YNeigh - 1, XNeigh - 1), (height, width))]

        if edge_type == 'regular':
            idxDiff = np.where(np.any(LCenter != LNeigh))[0]
        elif edge_type == 'inner':
            idxDiff = np.where(
                np.any(LCenter != LNeigh) &
                np.any(LCenter != 0) &
                np.all(LNeigh == 0)
            )[0]
        elif edge_type == 'outer':
            idxDiff = np.where(
                np.any(LCenter != LNeigh) &
                np.all(LCenter == 0) &
                np.any(LNeigh != 0)
            )[0]
        else:
            raise ValueError("Wrong edge type input!")

        idxIgnore2 = np.zeros(len(idxDiff), dtype=bool)
        idxDiffGT = idxDiff[~idxIgnore2]
        idxEdge[idxValid[idxDiffGT]] = True

    idxEdge = idxEdge.reshape((height, width))
    return idxEdge


def plot_predictions_and_input(data_folder: str, segmented_path: str, dataset: str = 'soda', split: str = 'test'):
    """
    Save plots of the input image with ground truth if exists and the predicted segmented image.
    Args:
        data_folder: path to the folder the processed data  is stored at.
        segmented_path: path to the folder the predicted segmented images are stored at.
        dataset: dataset name we are using.
        split: which split of data the segmentation is on, e.g. train, val, test.
    """
    image_folder = os.path.join(data_folder, 'image', split)
    mask_color_folder = os.path.join(data_folder, 'mask_color', split)
    mask_folder = os.path.join(data_folder, 'mask', split)
    images = sorted(f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f)))

    segmented_path = os.path.join(segmented_path, split)
    seg_preds = sorted(f for f in os.listdir(segmented_path)
                       if os.path.isfile(os.path.join(segmented_path, f)) and 'Pred' in str(f))
    edges_preds = sorted(f for f in os.listdir(segmented_path)
                         if os.path.isfile(os.path.join(segmented_path, f)) and 'Edges' in str(f))

    assert len(seg_preds) == len(images)
    save_dir = os.path.join('predictions_analysis', dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.isdir(mask_color_folder):  # if we had ground truth
        subplots = 4
        masks = sorted(f for f in os.listdir(mask_color_folder) if os.path.isfile(os.path.join(mask_color_folder, f)))
    elif os.path.isdir(mask_folder):
        subplots = 4
        masks = sorted(f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f)))
    else:
        subplots = 3
        masks = None

    for i in tqdm(range(len(images))):
        current_subplot = 1
        # saving all in one fig
        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        fig.tight_layout()
        ax[0, 0].imshow(Image.open(os.path.join(image_folder, images[i])), cmap='gray')
        ax[0, 0].set_title('Input image')
        current_subplot += 1

        ax[0, 1].imshow(Image.open(os.path.join(segmented_path, seg_preds[i])))
        ax[0, 1].set_title('Segmentation prediction')
        current_subplot += 1

        ax[1, 0].imshow(Image.open(os.path.join(segmented_path, edges_preds[i])))
        ax[1, 0].set_title('Edges prediction')
        current_subplot += 1

        if subplots == 4:
            ax[1, 1].imshow(Image.open(os.path.join(mask_color_folder, masks[i])))
            ax[1, 1].set_title('Ground truth')
            current_subplot += 1

        # saving fig
        plt.savefig(os.path.join(save_dir, f"predictions_analysis_{images[i]}"))
        plt.close(fig)


if __name__ == "__main__":
    # args = Namespace(
    #     # Models
    #     mode='test', train_only=False, model='ftnet', backbone='resnext50_32x4d', pretrained_base=False,
    #     dilation=False, pretrain_checkpoint=None,
    #     # Data and Dataloader
    #     dataset='soda', dataset_path='/home/ilan/Desktop/Niv/ThermalSegmentation/processed_dataset/SODA',
    #     base_size=[640], crop_size=[480], workers=16, no_of_filters=128, edge_extracts=[3], num_blocks=2,
    #     train_batch_size=16, val_batch_size=4, test_batch_size=1, accumulate_grad_batches=1, test_monitor='val_mIOU',
    #     test_monitor_path='/home/ilan/Desktop/Niv/ThermalSegmentation/soda//ckpt/',
    #     # WandB
    #     wandb_id=None, wandb_name_ext='None',
    #     # Training hyper params
    #     epochs=100, loss_weight=20,
    #     # Optimizer and scheduler parameters
    #     optimizer='SGD', lr=0.01, momentum=0.9, nesterov=False, weight_decay=0.0001, beta1=0.9, beta2=0.999,
    #     epsilon=1e-08, scheduler_type='poly_warmstartup', warmup_iters=0, warmup_factor=0.3333333333333333,
    #     warmup_method='linear', gamma=0.5,
    #     # Checkpoint and log
    #     resume=None, save_dir='/home/ilan/Desktop/Niv/ThermalSegmentation/soda//Best_MIOU/', test_checkpoint=None,
    #     save_images=True, save_images_as_subplots=False,
    #     # MISC
    #     debug=False, seed=123, num_nodes=1, gpus=1, distributed_backend='dp',
    # )
    # resnext50_32x4d
    args = Namespace(mode='test', train_only=False, model='ftnet', backbone='resnext101_32x8d', pretrained_base=False,
                     pretrain_checkpoint=None,
                     dilation=False, dataset='soda',
                     dataset_path='/home/ilan/Desktop/Niv/ThermalSegmentation/processed_dataset/SODA', base_size=None,
                     crop_size=None, workers=16, no_of_filters=128, edge_extracts=[3], num_blocks=2,
                     train_batch_size=4, val_batch_size=4, test_batch_size=1, accumulate_grad_batches=1,
                     test_monitor='val_mIOU', test_monitor_path='soda//ckpt/', wandb_id='m7mcxqut', wandb_name_ext='',
                     epochs=100, loss_weight=20, optimizer='SGD', lr=0.001, momentum=0.9, nesterov=False,
                     weight_decay=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08, scheduler_type='poly_warmstartup',
                     warmup_iters=0, warmup_factor=0.3333, warmup_method='linear', gamma=0.5,
                     resume=None,
                     save_dir='soda/', test_checkpoint=None,
                     save_images=True, save_images_as_subplots=False, debug=False, seed=0, num_nodes=1, gpus=1,
                     distributed_backend='ddp')
    main(args)
    # import cv2
    # seg = cv2.imread('processed_dataset/SODA/mask/train/train_00000034.png', cv2.IMREAD_GRAYSCALE)
    # seg = seg[:, :, None]  # adding channel dim
    # edgeMapBin = seg2edge(seg=seg, radius=1, edge_type='regular')  # Assuming seg2edge is defined
    # plt.imsave('processed_dataset/hi.png', edgeMapBin, cmap='gray')

    # current_path = os.getcwd()
    # plot_predictions_and_input(data_folder=os.path.join('processed_dataset/Tevel'),
    #                            segmented_path=os.path.join('tevel/Segmented_images'),
    #                            dataset='tevel')
    # print(os.listdir(os.path.join(current_path, 'tevel/Segmented_images/test')))

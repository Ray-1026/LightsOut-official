import argparse
import logging
import math
import os
import yaml
import torchvision
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.models.unet import U_Net
from src.models.light_regress_model import LightRegressModel
from utils.dataset import (
    Lightsource_Regress_Loader,
    Lightsource_3Maps_Loader,
    TestImageLoader,
)
from utils.loss import uncertainty_light_pos_loss, unet_3maps_loss
from utils.utils import IoU, mean_IoU


model_dict = {
    "unet": U_Net,
    "light_regress": LightRegressModel,  # this is the method in the paper
    "unet_3maps": U_Net,  # modified after submitted, more stable version
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        choices=model_dict.keys(),
        default="light_regress",
        help="Model to train",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/flare7kpp_dataset.yml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--save_ckpt_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training",
    )

    args = parser.parse_args()
    return args


class data_prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_batch["input"].cuda(non_blocking=True).float()
            self.next_target = self.next_batch["xyrs"].cuda(non_blocking=True).float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def collate_fn(batch, transform):
    # batch: list of dicts
    # transform: torchvision.transforms.Compose
    # return: transformed batch
    input_img = []
    control_img = []
    for data in batch:
        input_img.append(transform(data["input_img"]))
        control_img.append(transform(data["control_img"]))

    input_img = torch.stack(input_img, dim=0)
    control_img = torch.stack(control_img, dim=0)

    return {"input_img": input_img, "control_img": control_img}


class Trainer:
    def __init__(self, opt):
        self.opt = opt

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model_dict[self.opt.model]().to(self.device)

        if self.opt.model == "light_regress":
            self.criterion = uncertainty_light_pos_loss()
        elif self.opt.model == "unet_3maps":
            self.criterion = unet_3maps_loss()
        else:
            self.criterion = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=self.opt.lr,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.opt.epochs
        )

        with open(opt.config, "r") as stream:
            config = yaml.safe_load(stream)

        if self.opt.model == "light_regress":
            self.train_dataset = Lightsource_Regress_Loader(
                config["datasets"], num_lights=4
            )
        else:
            self.train_dataset = Lightsource_3Maps_Loader(
                config["datasets"], num_lights=4
            )

        self.eval_dataset = TestImageLoader(
            config["testing_dataset"]["dataroot_gt"],
            config["testing_dataset"]["dataroot_lq"],
            config["testing_dataset"]["dataroot_mask"],
            margin=0,
        )

        # test transformation
        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.opt.num_workers,
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, self.test_transform),
        )

        self.start_epoch = 0
        self.max_miou = 0.0

        # tensorboard
        self.writer = SummaryWriter(log_dir=os.path.join(opt.save_ckpt_dir, "logs"))

    def eval(self, epoch):
        self.model.eval()

        iou_scores = []
        with torch.no_grad():
            # for data in self.eval_dataset:
            for i, data in enumerate(self.test_loader):
                input_img = data["input_img"].to(self.device)
                control_img = data["control_img"]

                # input_img = self.test_transform(input_img).unsqueeze(0).to("cuda")
                # control_img = self.test_transform(control_img).unsqueeze(0)

                if self.opt.model == "light_regress":
                    pred_mask = self.model.forward_render(input_img)
                else:
                    pred_mask = self.model(input_img)

                if self.opt.model != "unet_3maps":
                    pred_mask = (pred_mask > 0.5).float()
                    pred_mask = pred_mask.cpu().numpy()
                else:
                    pred_prob, pred_rad = pred_mask[:, 0:1], pred_mask[:, 1:]
                    centers = self.extract_peaks(
                        pred_prob[0, 0]
                    )  # shape: pred_prob[0, 0] = (H, W)

                    if len(centers) == 0:
                        pred_mask = np.zeros_like(control_img[0, 0])
                    else:
                        radii = self.pick_radius(pred_rad[0, 0], centers)
                        pred_mask = self.draw_mask(centers, radii, 512, 512)

                control_img = (control_img.numpy() + 1) / 2

                iou_score = mean_IoU(control_img[0, 0], pred_mask[0, 0], 2)
                iou_scores.append(iou_score)

        mean_iou = np.mean(iou_scores)
        print(f"Mean IoU: {mean_iou:.4f}")

        self.writer.add_scalar("Eval/Mean_IoU", mean_iou, epoch)

        if mean_iou >= self.max_miou:
            self.max_miou = mean_iou
            self.save_checkpoint(self.model, self.optimizer, epoch)
        # elif epoch >= 20:
        #     self.save_checkpoint(self.model, self.optimizer, epoch)

        self.model.train()

    def train(self):
        torch.backends.cudnn.benchmark = True

        for epoch in range(self.start_epoch, self.opt.epochs):
            self.model.train()
            train_losses = []
            train_losses_xyr = []
            train_losses_p = []
            with tqdm(total=len(self.train_loader), unit="batch") as pbar:
                pbar.set_description(f"Epoch {epoch}")

                # for i, batch in enumerate(self.train_loader):
                i = 0
                prefetcher = data_prefetcher(self.train_loader)
                img, mask = prefetcher.next()
                while img is not None:
                    # img = batch["input"].to(self.device)
                    # mask = batch["light_masks_per"].to(self.device)

                    img = img.to(self.device)
                    mask = mask.to(self.device)

                    self.optimizer.zero_grad()

                    # output, _, _x = self.model(img)
                    output = self.model(img)

                    if self.opt.model == "unet":
                        loss = self.criterion(output, mask)
                    elif self.opt.model == "light_regress":
                        xyr_l, p_l = self.criterion(output, mask)
                        loss = xyr_l + p_l
                    else:
                        pred_prob, pred_rad = output[:, 0:1], output[:, 1:]
                        gt_prob, gt_rad = mask[:, 0:1], mask[:, 1:]

                        loss, L_prob, L_rad = self.criterion(
                            pred_prob, pred_rad, gt_prob, gt_rad
                        )

                    loss.backward()
                    self.optimizer.step()

                    train_losses.append(loss.item())
                    # train_losses_xyr.append(xyr_l.item())
                    # train_losses_p.append(p_l.item())
                    pbar.set_postfix(
                        # xyr_loss=f"{np.mean(train_losses_xyr):.4f}",
                        # p_loss=f"{np.mean(train_losses_p):.4f}",
                        loss=f"{np.mean(train_losses):.4f}",
                    )
                    pbar.update(1)

                    i += 1
                    img, mask = prefetcher.next()

                self.lr_scheduler.step()

                avg_loss = np.mean(train_losses)
                # avg_xyr_loss = np.mean(train_losses_xyr)
                # avg_p_loss = np.mean(train_losses_p)
                current_lr = self.optimizer.param_groups[0]["lr"]

                self.writer.add_scalar("Loss/Total", avg_loss, epoch)
                # self.writer.add_scalar("Loss/XYR_Loss", avg_xyr_loss, epoch)
                # self.writer.add_scalar("Loss/P_Loss", avg_p_loss, epoch)
                self.writer.add_scalar("Learning_Rate", current_lr, epoch)

                pbar.set_postfix(
                    # xyr_loss=f"{avg_xyr_loss:.4f}",
                    # p_loss=f"{avg_p_loss:.4f}",
                    loss=f"{avg_loss:.4f}",
                )
                pbar.close()

                self.eval(epoch)

        self.writer.close()

    def save_checkpoint(self, model, optimizer, epoch):
        # # remove previous checkpoints if ckpt more than 5
        # if len(os.listdir(self.opt.save_ckpt_dir)) >= 5:
        #     ckpts = os.listdir(self.opt.save_ckpt_dir)
        #     ckpts = [int(ckpt.split("_")[1].split(".")[0]) for ckpt in ckpts]
        #     ckpts.sort()
        #     os.remove(os.path.join(self.opt.save_ckpt_dir, f"model_{ckpts[0]}.pth"))

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            },
            os.path.join(self.opt.save_ckpt_dir, f"model_{epoch+1}.pth"),
        )

    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.checkpoint)
        self.model.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # epoch = checkpoint["epoch"]


def main():
    args = parse_args()
    os.makedirs(args.save_ckpt_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

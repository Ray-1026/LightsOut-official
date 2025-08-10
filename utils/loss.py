import torch
import torch.nn.functional as F
from torch import nn
from scipy.optimize import linear_sum_assignment


class uncertainty_light_pos_loss(nn.Module):
    def __init__(self):
        super(uncertainty_light_pos_loss, self).__init__()
        self.log_var_xyr = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.log_var_p = nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, logits, targets):
        B, N, P = logits.shape  # (B, 4, 4)

        position_loss = 0
        confidence_loss = 0

        w_xyr = 0.5 / (self.log_var_xyr**2)  # uncertainty weight for position loss
        w_p = 0.5 / (self.log_var_p**2)  # uncertainty weight for confidence loss
        weights = torch.tensor([1, 1, 2], device=logits.device)  # weights for x, y, r

        for b in range(B):
            pred_xyr = logits[b, :, :3]  # (N, 3)
            pred_p = logits[b, :, 3]  # (N,)

            gt_xyr = targets[b, :, :3]  # (N, 3)
            gt_p = targets[b, :, 3]  # (N,)

            cost_matrix = torch.cdist(gt_xyr, pred_xyr, p=2)  # (N, N)

            with torch.no_grad():
                row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())

            matched_pred_xyr = pred_xyr[col_ind]
            matched_gt_xyr = gt_xyr[row_ind]
            matched_pred_p = pred_p[col_ind]
            matched_gt_p = gt_p[row_ind]

            valid_mask = matched_gt_p > 0
            valid_cnt = valid_mask.sum().clamp(min=1)

            xyr_loss = (
                F.smooth_l1_loss(
                    matched_pred_xyr[valid_mask],
                    matched_gt_xyr[valid_mask],
                    reduction="none",
                )
                * weights
            ).sum()

            p_loss = F.binary_cross_entropy(
                matched_pred_p, matched_gt_p, reduction="mean"
            )

            position_loss += xyr_loss / valid_cnt
            confidence_loss += p_loss

        position_loss = w_xyr * (position_loss / B) + torch.log(1 + self.log_var_xyr**2)
        confidence_loss = w_p * (confidence_loss / B) + torch.log(1 + self.log_var_p**2)

        return position_loss, confidence_loss


class unet_3maps_loss(nn.Module):
    def __init__(self):
        super(unet_3maps_loss, self).__init__()

    def forward(self, pred_prob, pred_rad, prob_gt, rad_gt):
        focal = nn.BCELoss()
        L_prob = focal(pred_prob, prob_gt)

        pos_mask = prob_gt > 0.5
        L_rad = (
            nn.functional.smooth_l1_loss(pred_rad[pos_mask], rad_gt[pos_mask])
            if pos_mask.any()
            else pred_rad.sum() * 0
        )

        return L_prob + 10.0 * L_rad, L_prob, L_rad

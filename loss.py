import torch
from monai.losses import DiceCELoss
import numpy as np
from typing import Optional, Callable


def get_weights(n: int) -> list:
    unnormalised_weights = []
    w = 1
    for i in range(n):
        unnormalised_weights.append(w)
        w /= 2
    return (np.array(unnormalised_weights) / np.sum(unnormalised_weights)).tolist()


class DeepSuprDiceCELoss:

    '''
    This loss is suitable for use with monai.networks.DynUNet when deep supervision is turned on.
    '''

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        other_act: Optional[Callable] = None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
    ) -> None:
        
        self.LossFunction = DiceCELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            ce_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_ce=lambda_ce
        )

    def __call__(self, input: torch.Tensor, target: torch.Tensor):

        '''
        Inputs should be of shape (N, L, C, H, W, D) where N is batch dim, L is number of deep supervision outputs (L=0 is final layer output as
        is the case for DynUNet from monai.networks), C is number of classes, H, W, D are spatial dims. Targets should not have the L dim.
        '''
        deep_supr_num = input.size(1)
        deep_supr_weights = get_weights(deep_supr_num)
        input_tuple = torch.unbind(input, dim=1)

        loss = torch.tensor(0.0, device=input.device)
        for w, input_level in zip(deep_supr_weights, input_tuple):
            loss += w * self.LossFunction(input_level, target)

        return loss

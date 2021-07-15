import numpy as np
import torch
from loss_functions.focal_loss import FocalLoss
from torch import nn


class CB_loss(nn.Module):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """

    def __init__(self, samples_per_cls, no_of_classes, beta, gamma=0.):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        # in the official implementation, they normalize the weights based on inverse number of effective data per class
        weights = weights / np.sum(weights) * no_of_classes
        print(weights)
        weights = torch.tensor(weights).to(device, dtype=torch.float32)

        # if gamma = 0 and alpha is not indicated this is exactly the same as normal cross entropy with CB
        self.loss_fn = FocalLoss(alpha=weights, gamma=gamma)

    def forward(self, logits, labels):
        return self.loss_fn(logits, labels)


if __name__ == '__main__':
    no_of_classes = 2
    class_idx = 5
    logits = torch.tensor([[0.4163, 0.6000],
        [0.0287, 0.2432],
        [0.8611, 0.1739],
        [0.9699, 0.6336]])
    labels = torch.tensor([1, 1, 1, 0])
    beta = 0.99999
    gamma = 2
    samples_per_cls = [30974,54003,3566,27034,4475,669983,5260]
    samples_per_cls = [sum(samples_per_cls[:class_idx]) + sum(samples_per_cls[class_idx + 1:]), samples_per_cls[class_idx]]
    print(samples_per_cls)
    cb_loss = CB_loss(samples_per_cls, no_of_classes, beta, gamma)
    print(cb_loss(logits, labels))
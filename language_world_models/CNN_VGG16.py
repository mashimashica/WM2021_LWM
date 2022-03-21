import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

def vgg_based_model(use_gpu_number=0):
    """"""
    model = models.vgg16()

    #これで出力の次元数も変更できる
    # model.classifier.add_module('6', nn.Linear(4096, 2048))
    # model.classifier.add_module('7', nn.ReLU())
    # model.classifier.add_module('8', nn.Linear(2048, 1024))
    # model.classifier.add_module('9', nn.ReLU())
    # model.classifier.add_module('10', nn.Linear(1024, 1))

    model.cuda(use_gpu_number)

    return model
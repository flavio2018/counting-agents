import torch
from torch import nn

def initialize_weights(layer):
    """Initialize a layer's weights and biases.

    Args:
        layer: A PyTorch Module's layer."""
    if isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d)):
        pass
    else:
        try:
            nn.init.xavier_normal_(layer.weight)
        except AttributeError:
            pass
        try:
            nn.init.uniform_(layer.bias)
        except (ValueError, AttributeError):
            pass 


def concat_imgs_h(img_list):
    total_img = img_list[0]
    for img_i in img_list:
        total_img = concat_2_imgs_h(total_img, img_i)
    return total_img

def concat_imgs_v(img_list):
    total_img = img_list[0]
    for img_i in img_list:
        total_img = concat_2_imgs_v(total_img, img_i)
    return total_img

def concat_2_imgs_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_2_imgs_v(im1, im2):
    dst = Image.new('RGB', (im2.width, im1.height + im2.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
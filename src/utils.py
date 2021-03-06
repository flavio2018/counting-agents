"""
This file contains some utility functions used in different classes.
"""

from prettytable import PrettyTable

import numpy as np
import torch

from torch import nn
from PIL import Image, ImageDraw
from src import QLearning
#from fonts.ttf import AmaticSC


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


def concat_imgs_h(img_list, dist=0):
    total_img = img_list[0]
    for i in range(1, len(img_list)):
        total_img = concat_2_imgs_h(total_img, img_list[i], dist=dist)
    return total_img

def concat_imgs_v(img_list, dist=0):
    total_img = img_list[0]
    for i in range(1, len(img_list)):
        total_img = concat_2_imgs_v(total_img, img_list[i], dist=dist)
    return total_img

def concat_2_imgs_h(im1, im2, dist=0):
    dst = Image.new('RGB', (im1.width + dist + im2.width, im1.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width + dist, 0))
    return dst

def concat_2_imgs_v(im1, im2, dist=0):
    dst = Image.new('RGB', (im2.width, im1.height + dist + im2.height), color='white')
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + dist))
    return dst

def annotate_below(imgy, text):
    text_img = Image.new("RGBA", (int(imgy.width), int(imgy.height/4)), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    # font = ImageFont.load("arial.ttf")
    text_draw.text((0, 0), text, fill=0)
    imgy = concat_imgs_v([imgy, text_img])
    return imgy

def annotate_nodes(imgy, text_list):
    n_nodes = len(text_list)
    text_img = Image.new("RGBA", (int(2*imgy.width), int(imgy.height)), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    for n in range(n_nodes):
        # font = ImageFont.load("arial.ttf")
        text_draw.text((0, int((n+0.5)*imgy.height/len(text_list))), text_list[n], fill=0)
    imgy = concat_imgs_h([imgy, text_img])
    return imgy

def add_grid_lines(_img, _array):
    img_width = _img.width
    img_height = _img.height
    array_width = _array[0, :].size
    array_height = _array[:, 0].size
    draw = ImageDraw.Draw(_img)
    linewidth = 2
    for x in range(array_width + 1):
        line = ((x * img_width / array_width - int(linewidth / 2), 0),
                (x * img_width / array_width - int(linewidth / 2), img_height))
        draw.line(line, fill=0, width=2)
    for y in range(array_height + 1):
        line = ((0, y * img_height / array_height - int(linewidth / 2)),
                (img_width, y * img_height / array_height - int(linewidth / 2)))
        draw.line(line, fill=0, width=2)
    return _img


def test_agent(agent, env, n_runs):
    t = PrettyTable()
    t.field_names = ['# objects', 'avg accuracy']

    env.generate_random_nobj = False

    for n_objects in range(1, env.max_CL_objects+1):
        env.max_episode_objects = n_objects

        correct_label_bools = []

        for n_run in range(n_runs):
            env.reset()
            done = False

            while not done:
                state = torch.Tensor(env.state)
                q_values = QLearning.get_qvalues(state, agent)
                __, __, done, correct_label = env.step(
                    q_values,
                    env.max_train_iters,
                    state_visit_history
                )

            if correct_label:
                correct_label_bools.append(1)
            else:
                correct_label_bools.append(0)

        t.add_row([n_objects, np.mean(correct_label_bools)])

    print(t)
import torch
import itertools
import matplotlib.pyplot as plt
from config.config import *
import numpy as np
import os
import colormath
import colormath.color_conversions as conv
import yaml

def plot_color_association_bar_chart(color_list, hist_values, title):
    plt.title(title)
    plt.bar(x=range(len(hist_values)), height=hist_values, color=color_list)
    plt.show()


def plot_color_association_comparison(color_list, hist_values, titles, word, save_file_name=None):
    num_plots = len(hist_values)
    # Initialise the subplot function using number of rows and columns
    figure, axis = plt.subplots(num_plots)

    for i in range(num_plots):
        axis[i].bar(x=range(len(hist_values[i])), height=hist_values[i], color=color_list, edgecolor="black", linewidth=1)
        axis[i].set_title(titles[i])
        axis[i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
        axis[i].set_ylim(0, 0.5)
    figure.suptitle(word)
    if save_file_name != None:
        plt.savefig(save_file_name)
        plt.clf()
    else:
        plt.show()


# def get_rainbow_list(gradients_per_channel=range(255)):
#     color_list = torch.zeros([len(gradients_per_channel)**3, 3])
#     index = 0
#     for color in itertools.product(gradients_per_channel, repeat=3):
#         color_list[index] = torch.tensor(color) / 255
#         index += 1
#     return color_list


# def get_colors(color):
#     if color == "uw58":
#         colors_csv = np.loadtxt(os.path.join(ROOT, 'data', 'UW58_Colors.csv'), delimiter=",", dtype=str)[1:] # remove first (label) row
#         color_list = torch.zeros([58, 3])
#         for i in range(58):
#             rgb_color = conv.convert_color(colormath.color_objects.LabColor(lab_l=colors_csv[i, 3], lab_a=colors_csv[i, 4], lab_b=colors_csv[i, 5]), colormath.color_objects.sRGBColor)
#             color_list[i] = torch.tensor([rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b])
#         return color_list.unsqueeze(1).unsqueeze(1).repeat(1, 10, 10, 1)
    
#     elif type(color) == type(range(2)):
#         color_list = torch.zeros([len(color)**3, 3])
#         index = 0
#         for color in itertools.product(color, repeat=3):
#             color_list[index] = torch.tensor(color) / 255
#             index += 1
#         return color_list.type(torch.float).unsqueeze(1).unsqueeze(1).repeat(1, 10, 10, 1)
    
#     else:
#         raise Exception("unknown color argument passed to 'get_colors'")
    

# def get_concept_list(concepts):
#     with open(os.path.join(ROOT, 'experiments', 'concept_list.yml'), 'r') as file:
#         templates = yaml.safe_load(file)
#     if concepts in templates.keys():
#         return templates[concepts]
#     elif concepts == "all":
#         all_templates = []
#         for val in templates.values():
#             for i in val:
#                 if i not in all_templates:
#                     all_templates.append(i)
#         return all_templates
#     else:
#         raise Exception("given concept is not in concept_list.yml and is not 'all'")
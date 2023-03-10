import yaml
import os
from config.config import *
import numpy as np
import colormath
import colormath.color_conversions as conv
import itertools
from PIL import Image
import torchvision.transforms as transforms


def load_templates(template):
    with open(os.path.join(ROOT, 'data', 'templates.yml'), 'r') as file:
        templates = yaml.safe_load(file)
    if template in templates.keys():
        return templates[template]
    elif template == "all":
        all_templates = []
        for val in templates.values():
            for i in val:
                if i not in all_templates:
                    all_templates.append(i)
        return all_templates
    else:
        raise Exception("given template is not in templates.yml and is not 'all'")
    
def check_logits_are_probabilities(logits):
    assert 0.99999 <= sum(logits).item() <= 1.00001

def load_human_ratings(word, dataset="mukherjee"):

    if dataset == "rathore":
        # Participant, Concept, Color, Rating
        human_study_csv = np.loadtxt(os.path.join(ROOT, 'data', 'HumanRatingsData.csv'), delimiter=",", dtype=str)[1:] # remove first (label) row
        
        # filter to just rows with word
        mask = human_study_csv[:, 1] == word
        human_study_csv = human_study_csv[mask, :]

        ratings = []
        for participant in set(human_study_csv[:, 0]):
            mask = human_study_csv[:, 0] == participant
            individual = human_study_csv[mask, 3].astype(np.float)
            ratings.append(torch.tensor(individual))

        logits = torch.stack(ratings)
        # logits = torch.mean(logits, dim=0).softmax(dim=0)
        logits = torch.mean(logits, dim=0)
        logits = logits - torch.min(logits)
        logits = logits / torch.sum(logits)
        
    elif dataset == "mukherjee":
        # concept, color1_association, color2_association, ...
        human_study_csv = np.loadtxt(os.path.join(ROOT, 'data', 'uw_71_ratings_matrix.csv'), delimiter=",", dtype=str)[1:] # remove first (label) row

        concept_index = human_study_csv[:, 0] == '"' + word + '"'
        logits = torch.tensor(human_study_csv[concept_index, 1:].astype(np.float))

        # logits = torch.mean(logits, dim=0).softmax(dim=0)
        logits = torch.mean(logits, dim=0)
        logits = logits - torch.min(logits)
        logits = logits / torch.sum(logits)

    check_logits_are_probabilities(logits)
    return logits
    

def get_colors(color):
    if color == "uw58":
        colors_csv = np.loadtxt(os.path.join(ROOT, 'data', 'UW58_Colors.csv'), delimiter=",", dtype=str)[1:] # remove first (label) row
        color_list = torch.zeros([58, 3])
        for i in range(58):
            rgb_color = conv.convert_color(colormath.color_objects.LabColor(lab_l=colors_csv[i, 3], lab_a=colors_csv[i, 4], lab_b=colors_csv[i, 5]), colormath.color_objects.sRGBColor)
            color_list[i] = torch.tensor([rgb_color.clamped_rgb_r, rgb_color.clamped_rgb_g, rgb_color.clamped_rgb_b])
        return color_list.unsqueeze(1).unsqueeze(1).repeat(1, 10, 10, 1)
    
    elif color == "uw71":
        colors_csv = np.loadtxt(os.path.join(ROOT, 'data', 'UW71rgb.csv'), delimiter=",", dtype=float)
        color_list = torch.tensor(colors_csv)
        return color_list.unsqueeze(1).unsqueeze(1).repeat(1, 10, 10, 1)
    
    elif type(color) == type(range(2)):
        color_list = torch.zeros([len(color)**3, 3])
        index = 0
        for color in itertools.product(color, repeat=3):
            color_list[index] = torch.tensor(color) / 255
            index += 1
        return color_list.type(torch.float).unsqueeze(1).unsqueeze(1).repeat(1, 10, 10, 1)
    
    else:
        raise Exception("unknown color argument passed to 'get_colors'")
    

def get_concept_list(concepts):
    with open(os.path.join(ROOT, 'data', 'concept_list.yml'), 'r') as file:
        templates = yaml.safe_load(file)
    if concepts in templates.keys():
        return templates[concepts]
    elif concepts == "all":
        all_templates = []
        for val in templates.values():
            for i in val:
                if i not in all_templates:
                    all_templates.append(i)
        return all_templates
    else:
        raise Exception("given concept is not in concept_list.yml and is not 'all'")
    


def load_test_images(image_path):
    image_names = os.listdir(image_path)
    images = []
    # Define a transform to convert the image to tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([100, 100])])
    for image in image_names:
        # Read the image
        img = Image.open(os.path.join(image_path, image))
        # Convert the image to PyTorch tensor
        tensor_img = transform(img)
        images.append(tensor_img)
    return torch.stack(images), image_names
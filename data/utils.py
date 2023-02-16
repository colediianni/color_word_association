import yaml
import os
from config.config import *
import numpy as np

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
    
def load_human_ratings(word):
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

    logits = torch.stack(ratings).to(device)
    # logits = torch.mean(logits, dim=0).softmax(dim=0)
    logits = torch.mean(logits, dim=0)
    logits = logits - torch.min(logits)
    logits = logits / torch.mean(logits)

    return logits
    
import torch

ROOT = r"C:\Users\Cole\Desktop\color_word_association"
DATASETS = ["mukherjee", "rathore"]
MODELS = ["clip"]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEMPLATES = ["cifar100", "caltech101", "custom", "foods", "food_and_concepts", "all"]
DATASET_COLOR_DICT = {"mukherjee": "uw71", "rathore": "uw58"}
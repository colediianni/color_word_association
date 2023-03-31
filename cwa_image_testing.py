from models.utils import load_model, get_color_word_associations
from data.utils import get_concept_list, load_templates, load_human_ratings, get_colors, load_test_images
from experiments.utils import plot_color_association_bar_chart
from config.config import DATASET_COLOR_DICT, ROOT
from scipy import stats
from config.parser import parser
import matplotlib.pyplot as plt
import os
import torch
import copy


def main():
    args = parser()
    model, processor = load_model(args.model_type)
    prompts = load_templates(args.template)
    test_images, image_names = load_test_images(os.path.join(ROOT, 'data', "images"))
    image_names = image_names + ["swapped_"+name for name in image_names]
    swapped_images = copy.deepcopy(test_images)
    # print(swapped_images.shape)
    swapped_images[:, 0, :, :] = test_images[:, 1, :, :]
    swapped_images[:, 1, :, :] = test_images[:, 0, :, :]
    test_images = torch.cat([test_images, swapped_images])
    test_images = test_images.to(args.device)

    output_folder = os.path.join(ROOT, 'output', "apple_visualizations")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    concept = "apple"

    for test_image in range(test_images.shape[0]):
        # print(test_images[test_image].shape)
        model_logits = get_color_word_associations(concept, prompts, test_images[test_image], processor, model, convert_to_prob=False)
        print(image_names[test_image])
        print(model_logits)
        save_path = os.path.join(output_folder, str(model_logits.item()) + image_names[test_image])
        plt.imshow(test_images[test_image].detach().permute([1, 2, 0]).cpu().numpy())
        plt.savefig(save_path)
        plt.clf()

if __name__ == "__main__":
    main()
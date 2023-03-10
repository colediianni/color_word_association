from models.utils import load_model, get_color_word_associations
from data.utils import get_concept_list, load_templates, load_human_ratings, get_colors, load_test_images
from experiments.utils import plot_color_association_bar_chart
from config.config import DATASET_COLOR_DICT, ROOT
from scipy import stats
from config.parser import parser
import os


def main():
    args = parser()
    model, processor = load_model(args.model_type)
    prompts = load_templates(args.template)
    test_images, image_names = load_test_images(os.path.join(ROOT, 'data', "images"))
    test_images = test_images.to(args.device)

    # output_folder = os.path.join(ROOT, 'output', "test_visualizations")

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    concept = "apple"

    for test_image in range(test_images.shape[0]):
        model_logits = get_color_word_associations(concept, prompts, test_images[test_image], processor, model, convert_to_prob=False)
        print(image_names[test_image])
        print(model_logits)

if __name__ == "__main__":
    main()
from models.utils import load_model, get_color_word_associations
from data.utils import get_concept_list, load_templates, load_human_ratings, get_colors
from experiments.utils import plot_color_association_bar_chart
from config.config import DATASET_COLOR_DICT, ROOT
from scipy import stats
from config.parser import parser
import os


def main():
    args = parser()
    model, processor = load_model(args.model_type)
    prompts = load_templates(args.template)
    image = get_colors(DATASET_COLOR_DICT[args.dataset]).to(args.device) # "uw58"
    print(image.shape[0], "unique colors")

    colors = [image[sample].permute(2, 0, 1) for sample in range(image.shape[0])]
    output_folder = os.path.join(ROOT, 'output', "test_visualizations")

    if not  os.path.exists(output_folder):
        os.makedirs(output_folder)

    for concept in ["red", "green", "blue", "purple", "black", "white", "pink", "yellow", "orange"]:
        print(concept)
        # human_logits = load_human_ratings(concept, args.dataset).cpu()
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], human_logits, concept)

        model_logits = get_color_word_associations(concept, prompts, colors, processor, model)
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], model_logits.squeeze(), concept)

        save_path = os.path.join(output_folder, concept+".png")
        plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], model_logits, concept, save_file_name=save_path)


if __name__ == "__main__":
    main()
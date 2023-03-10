from models.utils import load_model, get_color_word_associations
from data.utils import get_concept_list, load_templates, load_human_ratings, get_colors
from experiments.utils import plot_color_association_comparison
from config.config import DATASET_COLOR_DICT, ROOT
from scipy import stats
from config.parser import parser
import os


def main():
    args = parser()
    model, processor = load_model(args.model_type)
    prompts = load_templates(args.template) # "custom"
    image = get_colors(DATASET_COLOR_DICT[args.dataset]).to(args.device) # "uw58"
    print(image.shape[0], "unique colors")

    colors = [image[sample].permute(2, 0, 1) for sample in range(image.shape[0])]
    concept_list = get_concept_list(args.dataset)
    # output_folder = os.path.join(ROOT, 'output', args.dataset+"_visualizations")

    # if not  os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    corrCoeffList = []
    wasserstein_dist_list = []
    for concept in concept_list:
        print(concept)
        human_logits = load_human_ratings(concept, args.dataset).cpu()
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], human_logits, concept)

        model_logits = get_color_word_associations(concept, prompts, colors, processor, model)
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], model_logits.squeeze(), concept)

        # save_path = os.path.join(output_folder, concept+".png")
        # plot_color_association_comparison([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], [human_logits, model_logits], ["human", args.model_type], concept, save_file_name=save_path)

        wasserstein_dist = stats.wasserstein_distance(human_logits, model_logits)
        print("wasserstein dist: ", wasserstein_dist)
        wasserstein_dist_list.append(wasserstein_dist)

        corrCoeff, pVal = stats.pearsonr(human_logits, model_logits)
        corrCoeffList.append(corrCoeff)
        print("\nPearson’s correlation coefficient (between true ratings and avg predicted ratings) is %.4f" %corrCoeff)
        print("p-value %.4f" %pVal)

    print("average wasserstein dist", sum(wasserstein_dist_list) / len(wasserstein_dist_list))
    print("average corr coef", sum(corrCoeffList) / len(corrCoeffList))


if __name__ == "__main__":
    main()

# mukherjee: average wasserstein dist 0.00591344604318539
# rathore: average wasserstein dist 0.01233944862472298


# softmax both
# custom average corr coef 0.6618853608719082
# cifar100 average corr coef 0.651793663869699

# (logit - min) / average
# custom average corr coef 0.7655195774843114
# cifar100 average corr coef 0.7132087811258802

# mix
# custom average corr coef 0.6487777991562128
# cifar100 average corr coef 0.641209737035525

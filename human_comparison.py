from models.utils import *
from data.utils import *
from experiments.utils import *
from config.config import *
from scipy import stats

def main():
    model, processor = load_model("clip")
    prompts = load_templates("custom")
    image = get_colors("uw58")
    print(image.shape[0], "unique colors")

    colors = [image[sample].permute(2, 0, 1) for sample in range(image.shape[0])]
    concept_list = get_concept_list("human_test")

    corrCoeffList = []
    for concept in concept_list:
        print(concept)
        human_logits = load_human_ratings(concept).cpu()
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], human_logits, concept)

        model_logits = get_color_word_associations(concept, prompts, colors, processor, model)
        # plot_color_association_bar_chart([image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])], model_logits.squeeze(), concept)

        corrCoeff, pVal = stats.pearsonr(human_logits, model_logits)
        corrCoeffList.append(corrCoeff)
        print("\nPearson’s correlation coefficient (between true ratings and avg predicted ratings) is %.4f" %corrCoeff)
        print("p-value %.4f" %pVal)

    print("average corr coef", sum(corrCoeffList) / len(corrCoeffList))


if __name__ == "__main__":
    main()

# softmax both
# custom average corr coef 0.6618853608719082
# cifar100 average corr coef 0.651793663869699

# (logit - min) / average
# custom average corr coef 0.7655195774843114
# cifar100 average corr coef 0.7132087811258802

# mix
# custom average corr coef 0.6487777991562128
# cifar100 average corr coef 0.641209737035525

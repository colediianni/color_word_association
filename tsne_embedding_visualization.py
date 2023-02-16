from models.utils import *
from data.utils import *
from experiments.utils import *
from config.config import *
import sklearn.manifold

def cs(x, y):
    x, y = torch.tensor(x), torch.tensor(y)
    return 1 - torch.nn.CosineSimilarity(dim=0, eps=1e-08)(x, y)


def main():
    image = get_colors(range(0, 255, 20)).type(torch.float).to(device)
    print(image.shape[0], "unique colors")
    colors = [image[sample].permute(2, 0, 1) for sample in range(image.shape[0])]
    prompts = load_templates("custom")
    model, processor = load_model("clip")

    encoded_image, text_embeddings = get_model_embeddings("test", prompts, colors, processor, model)
    # encoded_image = encoded_image.squeeze().permute([1, 0]).detach().cpu().numpy()

    tsne = sklearn.manifold.TSNE(n_components=3, verbose=1, random_state=123)
    z = tsne.fit_transform(encoded_image[0])

    # plt.scatter(z[:,0], z[:,1], c=[image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])])
    # plt.show()

    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    # Creating plot
    ax.scatter3D(z[:,0], z[:,1], z[:,2], color=[image[sample, 0, 0].cpu().numpy() for sample in range(image.shape[0])])
    plt.title("Color Embedding TSNE Visualization")
    # show plot
    plt.show()






if __name__ == "__main__":
    main()
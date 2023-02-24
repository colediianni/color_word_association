import torch
from transformers import CLIPProcessor, CLIPModel
import tqdm

def load_model(model_name):
    if model_name == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor


def check_logits_are_probabilities(logits):
    assert 0.99999 <= sum(logits).item() <= 1.00001

def get_color_word_associations(text, prompts, colors, processor, model):
    logits = []
    for prompt in tqdm.tqdm(prompts):
        with torch.no_grad():
            text_prompt = prompt.replace("{}", text)
            inputs = processor(text=text_prompt, images=colors, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            logits_per_text = outputs.logits_per_text #.softmax(dim=1)
            logits.append(logits_per_text)        
            
    logits = torch.cat(logits)
    logits = torch.mean(logits, dim=0).softmax(dim=0)
    # logits = torch.mean(logits, dim=0)
    # logits = logits - torch.min(logits)
    # logits = logits / torch.sum(logits)
    check_logits_are_probabilities(logits)
    return logits


def get_model_embeddings(text, prompts, colors, processor, model):
    image_embeddings, text_embeddings = [], []
    for prompt in tqdm.tqdm(prompts):
        with torch.no_grad():
            text_prompt = prompt.replace("{}", text)
            inputs = processor(text=text_prompt, images=colors, return_tensors="pt", padding=True)
            outputs = model(**inputs)

            image_embeddings.append(outputs.image_embeds)
            text_embeddings.append(outputs.text_embeds)
    
    return image_embeddings, text_embeddings
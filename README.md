# color_word_association

This repository explores the similarity between human and vision-language model color word associations. We pass solid color images into a CLIP model with words to test associations and find that the CLIP model's color word association distribution closely matches those of humans, with a suprising outlier of colors such as black, white, and grey. We hypothesize this may be due to the incorporation of greyscale as an augmentation in the CLIP model's training pipeline.

datasets used:
mukherjee - from "context matters: a theory of semantic discriminability for perceptual encodina systems" (more extensive word list using 71 colors)
rathore - from "estimating color-concept associations from image statistics" (basic fruits using 58 colors)

# Vision Tranformer for image classification

Image Classification with Vision Transformer for blog image selection

## How to use

### train
1. Set the folder location where the images are located for train (train.py: train_list [line. 30])
2. Rename image file or change method for get label (train.py: label_list [line. 31])
3. Run train.py

### inference
1. Set the folder location where the weight is located (inference.py: weight_path [line. 15])
2. Set the folder location where the images are located for inference (inference.py: inference_list [line. 21])
3. Run inference.py

## Requirements

- torch == 1.13.1+cu116
- torchaudio == 0.13.1+cu116
- torchvision == 0.14.1+cu116
- python == 3.6.9
- scikit-learn == 1.2.1
- Pillow == 9.4.0
- numpy == 1.24.2

## Reference

- [Vision Transformer and MLP-Mixer Architectures(Google[jax])](https://github.com/google-research/vision_transformer)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [ViT-L_16.npz](https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz)

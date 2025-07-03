# Image-Captioning

Generate captions for images using deep learning magic — no fluff, just results.

## What’s the Project?

A **Jupyter Notebook**‑powered pipeline that builds an image-captioning model using **CNNs + RNNs (or LSTMs)**. It takes visual features from images and spits out fun, relevant captions.

## High‑Level Overview

- **CNN encoder** extracts visual features (think InceptionV3/VGG/DenseNet).
- **RNN/LSTM decoder** takes those features to generate captions word-by-word.
- Inspired by "Show, Attend and Tell" and other OG models mixing CNN + LSTM.
- Trained on a dataset like **Flickr8k** (8k images, 5 captions each).

## What’s Inside the Notebook

1. **Data setup**  
   - Load dataset and parse image–caption pairs.  
   - Preprocess captions (tokenize, pad, etc.) and split train/val.

2. **Feature extraction**  
   - Use pre-trained CNN to pull out image embeddings (transfer learn).

3. **Text pipeline**  
   - Tokenizer → integer sequences → padded to fixed length.

4. **Decoder model**  
   - LSTM + Dense layers that decode image features + text context → generate captions.

5. **Training loop**  
   - Fit model, track loss, maybe BLEU metrics.

6. **Inference**  
   - Greedy or beam-search caption generation for test images.

## Requirements

- Python 3.x  
- TensorFlow + Keras  
- (Optional) Jupyter Notebook for execution  
- Numpy, tqdm, matplotlib (for data & visuals)

## How to run ?
git clone https://github.com/DarshMatariya/Image-captioning.git
cd Image-captioning
pip install -r requirements.txt
jupyter notebook  


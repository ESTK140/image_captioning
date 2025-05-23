# 🖼️ Image Captioning using ResNet50 + LSTM

This project implements an end-to-end deep learning pipeline to **generate natural language captions for images**, trained on the **Flickr8k dataset**. It uses **ResNet50** as an image encoder and **LSTM** as a caption decoder.

---

## 📌 Features

- Automatic download and extraction of Flickr8k image and text datasets
- Image preprocessing and feature extraction using pre-trained ResNet50
- Caption generation using LSTM-based model with embedding and attention
- Model checkpoint saving during training
- Demo prediction with visualization of a captioned image

---

## 📁 Project Structure

```bash
image_captioning/
│
├── train_and_predict.py     # Main script for training and demo prediction
├── tokenizer.pkl            # Tokenizer object for captioning
├── max_length.txt           # Max caption length used for padding
├── image_caption_model.h5   # Trained model (saved after training)
├── Flickr8k_Dataset.zip     # Image dataset (auto-downloaded)
├── Flickr8k_text.zip        # Caption dataset (auto-downloaded)
├── Flicker8k_Dataset/       # Extracted image folder
├── Flickr8k_text/           # Extracted caption folder
├── Model.H

model : https://drive.google.com/file/d/1yPGFZm64XxCsovgx6-OQxiKUaBirb4aF/view?usp=sharing

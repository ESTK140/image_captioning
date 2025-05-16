import os
import zipfile
import numpy as np
import string
import requests
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import random
import tensorflow as tf

print("\n🖥️ Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ Using GPU: Physical GPUs = {len(gpus)}, Logical GPUs = {len(logical_gpus)}")
    except RuntimeError as e:
        print("❌ GPU Setup Error:", e)
else:
    print("⚠️ No GPU found. Running on CPU.")

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint

def download_dataset():
    if not os.path.exists("Flickr8k_Dataset.zip"):
        print("Downloading Flickr8k images...")
        url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
        r = requests.get(url, stream=True)
        with open("Flickr8k_Dataset.zip", "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)

    if not os.path.exists("Flickr8k_text.zip"):
        print("Downloading Flickr8k captions...")
        url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
        r = requests.get(url, stream=True)
        with open("Flickr8k_text.zip", "wb") as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024)):
                if chunk:
                    f.write(chunk)

    if not os.path.exists("Flicker8k_Dataset"):
        with zipfile.ZipFile("Flickr8k_Dataset.zip", 'r') as zip_ref:
            zip_ref.extractall()
    if not os.path.exists("Flickr8k_text"):
        with zipfile.ZipFile("Flickr8k_text.zip", 'r') as zip_ref:
            zip_ref.extractall()

def load_captions(path):
    captions = {}
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split('\t')
            if len(tokens) != 2: continue
            img_id, caption = tokens
            img_id = img_id.split('.')[0]
            caption = 'startseq ' + caption.lower().translate(str.maketrans('', '', string.punctuation)) + ' endseq'
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    return captions

def extract_features(directory):
    model = ResNet50(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = {}
    for img_name in tqdm(os.listdir(directory)[:3000], desc="Extracting features"):
        path = os.path.join(directory, img_name)
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        features[img_name.split('.')[0]] = feature
    return features

def create_tokenizer(descriptions):
    lines = [c for cap in descriptions.values() for c in cap]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, desc, photo, vocab_size):
    X1, X2, y = [], [], []
    for caption in desc:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1, dtype='float16'), np.array(X2), np.array(y, dtype='float16')

def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), dtype='float32')  # ปรับ dtype
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_model():
    download_dataset()
    descriptions = load_captions("Flickr8k.token.txt")
    features = extract_features("Flicker8k_Dataset")
    tokenizer = create_tokenizer(descriptions)
    max_length = max(len(c.split()) for caps in descriptions.values() for c in caps)
    vocab_size = len(tokenizer.word_index) + 1

    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    with open('max_length.txt', 'w') as f:
        f.write(str(max_length))

    model = define_model(vocab_size, max_length)
    checkpoint = ModelCheckpoint('model_best.h5', monitor='loss', save_best_only=True)

    X1, X2, y = [], [], []
    for key in list(descriptions.keys())[:1000]:
        if key not in features:
            continue
        img_features = features[key][0]
        input_captions = random.sample(descriptions[key], k=min(2, len(descriptions[key])))
        a, b, c = create_sequences(tokenizer, max_length, input_captions, img_features, vocab_size)
        X1.append(a)
        X2.append(b)
        y.append(c)

    X1 = np.vstack(X1)
    X2 = np.vstack(X2).astype(np.float32)  # ปรับ dtype
    y = np.vstack(y)

    model.fit([X1, X2], y, epochs=10, verbose=2, callbacks=[checkpoint])
    model.save('image_caption_model.h5')

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        sequence = sequence.astype(np.float32)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = [k for k, v in tokenizer.word_index.items() if v == yhat]
        if not word:
            break
        word = word[0]
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def predict_demo():
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    with open('max_length.txt', 'r') as f:
        max_length = int(f.read())
    model = load_model('image_caption_model.h5')

    resnet = ResNet50(weights='imagenet')
    modelCNN = Model(resnet.input, resnet.layers[-2].output)

    test_img_path = "Flicker8k_Dataset/1000268201_693b08cb0e.jpg"
    image = load_img(test_img_path, target_size=(224, 224))
    photo = img_to_array(image)
    photo = np.expand_dims(photo, axis=0)
    photo = preprocess_input(photo)
    photo = modelCNN.predict(photo, verbose=0)

    caption = generate_caption(model, tokenizer, photo, max_length)
    print("Predicted:", caption)

    plt.imshow(Image.open(test_img_path))
    plt.title(caption.replace('startseq', '').replace('endseq', ''))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    train_model()
    predict_demo()

import os
import random
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# Load tokenizer and max_length
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
with open('max_length.txt', 'r') as f:
    max_length = int(f.read())

# Load trained captioning model
model = load_model('image_caption_model.h5')

# Load CNN encoder (ResNet50)
resnet = ResNet50(weights='imagenet')
modelCNN = Model(resnet.input, resnet.layers[-2].output)

# Function: Generate caption from photo feature
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

# 🔄 Pick a random image from the dataset
image_dir = "Flicker8k_Dataset"
random_img_name = random.choice(os.listdir(image_dir))
random_img_path = os.path.join(image_dir, random_img_name)

# 🖼️ Preprocess the selected image
image = load_img(random_img_path, target_size=(224, 224))
photo = img_to_array(image)
photo = np.expand_dims(photo, axis=0)
photo = preprocess_input(photo)
photo = modelCNN.predict(photo, verbose=0)

# 🧠 Generate caption
caption = generate_caption(model, tokenizer, photo, max_length)
caption_cleaned = caption.replace("startseq", "").replace("endseq", "").strip()

# 🖼️ Show image and caption
plt.imshow(Image.open(random_img_path))
plt.title(caption_cleaned)
plt.axis('off')
plt.show()

print(f"\n🖼️ File: {random_img_name}")
print(f"📝 Caption: {caption_cleaned}")

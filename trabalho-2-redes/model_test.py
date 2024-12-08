import tensorflow as tf
import os
from PIL import Image
import numpy as np

current_dir = os.getcwd()
experimento_dir = os.path.join(current_dir, 'experimento')

img_height = 128
img_width = 128

model = tf.keras.models.load_model('logo_classifier.keras')

experimento_images = []

classes = ['Athletico', 'Atletico_Goianiense', 'Atletico_Mineiro', 'Bahia', 'Botafogo', 'Bragantino', 'Corinthians', 'Criciuma', 'Cruzeiro', 'Cuiaba',
           'Flamengo', 'Fluminense', 'Fortaleza', 'Gremio', 'Internacional', 'Juventude', 'Palmeiras', 'Sao_Paulo', 'Vasco_da_Gama', 'Vitoria']

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  
    img = img.resize((img_height, img_width))
    # Normaliza a imagem
    img = np.array(img) / 255.0
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

for file in os.listdir(experimento_dir):
    experimento_images.append(load_and_preprocess_image(os.path.join(experimento_dir, file)))

experimento_images = np.concatenate(experimento_images, axis=0)

# Fazer a predição
predictions = model.predict(experimento_images)

for i, prediction in enumerate(predictions):
    print(f'Predição para {os.listdir(experimento_dir)[i]}: {classes[np.argmax(prediction)]}')
    print(f'Probabilidades: {prediction}')
    print()


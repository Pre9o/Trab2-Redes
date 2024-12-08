import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from PIL import Image


current_dir = os.getcwd()
train_dir = os.path.join(current_dir, 'augmented_logos', 'train')
test_dir = os.path.join(current_dir, 'augmented_logos', 'test')

classes = ['Athletico', 'Atletico_Goianiense', 'Atletico_Mineiro', 'Bahia', 'Botafogo', 'Bragantino', 'Corinthians', 'Criciuma', 'Cruzeiro', 'Cuiaba',
           'Flamengo', 'Fluminense', 'Fortaleza', 'Gremio', 'Internacional', 'Juventude', 'Palmeiras', 'Sao_Paulo', 'Vasco_da_Gama', 'Vitoria']

img_height = 128
img_width = 128

def remove_iccp_profile(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                filepath = os.path.join(root, filename)
                img = Image.open(filepath)
                if 'icc_profile' in img.info:
                    img.save(filepath, icc_profile=None)

remove_iccp_profile(train_dir)
remove_iccp_profile(test_dir)


train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=classes,
    color_mode='rgb',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='training'
)
print(train_ds)

val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=classes,
    color_mode='rgb',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset='validation'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    class_names=classes,
    color_mode='rgb',
    batch_size=32,
    image_size=(img_height, img_width),
    shuffle=True,
    seed=123
)

import matplotlib.pyplot as plt

def plot_images_with_labels(dataset, class_names, num_images=9):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(num_images):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.savefig('images.png')

plot_images_with_labels(train_ds, classes)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

kernel_size = (3, 3)
input_shape = (img_height, img_width, 3)


base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False 

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(classes), activation='softmax')
])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

batch_size = 32

METRICS = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]

learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])


class SavePredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, class_names, save_dir='predictions'):
        super(SavePredictionsCallback, self).__init__()
        self.dataset = dataset
        self.class_names = class_names
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        for images, labels in self.dataset.take(1):
            predictions = self.model.predict(images)
            # Desnormalizar as imagens
            images = images * 255
            plt.figure(figsize=(10, 10))
            for i in range(min(9, len(images))):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                predicted_class = self.class_names[np.argmax(predictions[i])]
                true_class = self.class_names[labels[i]]
                plt.title(f"Pred: {predicted_class}\nTrue: {true_class}")
                plt.axis("off")
            save_path = os.path.join(self.save_dir, f'epoch_{epoch + 1}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Predições salvas em {save_path}")

save_predictions_callback = SavePredictionsCallback(test_ds, classes)


steps_per_epoch = train_ds.cardinality().numpy() // batch_size

model.fit(train_ds, validation_data=val_ds, epochs=20, steps_per_epoch=steps_per_epoch, callbacks=[early_stopping, save_predictions_callback], batch_size=batch_size)


model.evaluate(test_ds)

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB') 
    img = img.resize((img_height, img_width))
    # Normaliza a imagem
    img = np.array(img) / 255.0
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

test_images = []
test_labels = []

for class_name in classes:
    test_image_path = os.path.join(test_dir, class_name, f'{class_name}_1800.png')
    test_images.append(load_and_preprocess_image(test_image_path))
    test_labels.append(classes.index(class_name))

test_images = np.concatenate(test_images, axis=0)
test_labels = np.array(test_labels)

predictions = model.predict(test_images)

correct_predictions = 0
for i, prediction in enumerate(predictions):
    if np.argmax(prediction) == test_labels[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(predictions)
print(f'Acurácia: {accuracy}')

for i, prediction in enumerate(predictions):
    print(f'Predição: {classes[np.argmax(prediction)]} - Label: {classes[test_labels[i]]}')


teste1_path = os.path.join(current_dir, 'teste1.jpg')
teste2_path = os.path.join(current_dir, 'teste2.png')
teste3_path = os.path.join(current_dir, 'teste3.jpg')
teste4_path = os.path.join(current_dir, 'teste4.png')

teste1 = load_and_preprocess_image(teste1_path)
teste2 = load_and_preprocess_image(teste2_path)
teste3 = load_and_preprocess_image(teste3_path)
teste4 = load_and_preprocess_image(teste4_path)

testes = [teste1, teste2, teste3, teste4]

for i, teste in enumerate(testes):
    prediction = model.predict(teste)
    print(f'Predição para teste{i + 1}: {classes[np.argmax(prediction)]}')

# Salvar o modelo
model.save('logo_classifier.keras')



import os
import Augmentor
from PIL import Image

current_dir = os.getcwd()
original_logos_dir = os.path.join(current_dir, 'original_logos')

resized_logos_dir = os.path.join(current_dir, 'resized_logos')
augmented_logos_dir = os.path.join(current_dir, 'augmented_logos')
os.makedirs(resized_logos_dir, exist_ok=True)
os.makedirs(augmented_logos_dir, exist_ok=True)

classes = ['Athletico', 'Atletico_Goianiense', 'Atletico_Mineiro', 'Bahia', 'Botafogo', 'Bragantino', 'Corinthians', 'Criciuma', 'Cruzeiro', 'Cuiaba',
           'Flamengo', 'Fluminense', 'Fortaleza', 'Gremio', 'Internacional', 'Juventude', 'Palmeiras', 'Sao_Paulo', 'Vasco_da_Gama', 'Vitoria']

def resize_images():
    for logo in os.listdir(original_logos_dir):
        logo_path = os.path.join(original_logos_dir, logo)
        img = Image.open(logo_path)
        img = img.resize((128, 128))
        
        resized_logo_path = os.path.join(resized_logos_dir, logo)
        img.save(resized_logo_path)

def augment_logos(num_variations=20000):       
    p = Augmentor.Pipeline(resized_logos_dir, output_directory=augmented_logos_dir)
    p.ground_truth(resized_logos_dir)
    p.rotate(probability=0.7, max_left_rotation=25, max_right_rotation=25)
    p.flip_left_right(probability=0.5)
    p.flip_top_bottom(probability=0.5)
    p.zoom_random(probability=0.5, percentage_area=0.8)
    p.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_color(probability=0.5, min_factor=0.7, max_factor=1.3)
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)
    
    p.sample(num_variations)

def create_classes():
    for class_name in classes:
        os.makedirs(os.path.join(augmented_logos_dir, class_name), exist_ok=True)

def label_images():
    logos = os.listdir(augmented_logos_dir)
    for logo in logos:
        if '.png' not in logo:
            continue
        for class_name in classes:
            if class_name in logo:
                os.rename(os.path.join(augmented_logos_dir, logo), os.path.join(augmented_logos_dir, class_name, logo))
                break

def split_train_test():
    for class_name in classes:
        images = os.listdir(os.path.join(augmented_logos_dir, class_name))
        train_dir = os.path.join(augmented_logos_dir, 'train', class_name)
        test_dir = os.path.join(augmented_logos_dir, 'test', class_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        for i, image in enumerate(images):
            if i < len(images) * 0.8:
                os.rename(os.path.join(augmented_logos_dir, class_name, image), os.path.join(train_dir, f'{class_name}_{i}.png'))
            else:
                os.rename(os.path.join(augmented_logos_dir, class_name, image), os.path.join(test_dir, f'{class_name}_{i}.png'))


resize_images()
augment_logos()
create_classes()
label_images()
split_train_test()
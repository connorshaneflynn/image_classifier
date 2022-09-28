import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import json

parser = argparse.ArgumentParser()

parser.add_argument('input_image', action='store', type=str, help='Path to input image')
parser.add_argument('classifier', action='store', type=str, help='Path to classifier')
parser.add_argument('--top_k', default=3, action='store', type=int, help='Gives back the top K most likely classes')
parser.add_argument('--category_names', default='label_map.json', action='store', type=str, help='Label mapping (.json)') # leave default as empty string for no mapping, only prints labels

arg_parser = parser.parse_args()
top_k = arg_parser.top_k


def process_image(image):
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    return image


def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)
    
    values, indices = tf.nn.top_k(pred, k=top_k)
    probabilities = list(values.numpy()[0])
    classes = list(indices.numpy()[0])
                   
    return probabilities, classes

if arg_parser.category_names:
    with open(arg_parser.category_names, 'r') as f:
        mapping = json.load(f)
    
model = tf.keras.models.load_model(arg_parser.classifier, compile=False, custom_objects = {'KerasLayer':hub.KerasLayer})

probabilities, labels = predict(arg_parser.input_image, model, top_k)

print(f'\n\nThe top {top_k} Classes are:\n')
for probability, label in zip(probabilities, labels):
    if arg_parser.category_names:
        print('Class:\t', mapping[str(label+1)].title())
    else:
        print('Label:\t', str(label))
        
    print('  ', probability,'%\n')
                  

    
  










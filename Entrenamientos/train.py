import numpy as np
import tensorflow as tf
import json

from utils import util
from transformers.models.segformer import modeling_tf_segformer as models



# Importación de datos
dir_img = "./cityscapes/images"
dir_lab = "./cityscapes/labels"

img_dataset_train = tf.keras.utils.image_dataset_from_directory(dir_img, batch_size=5, labels = None, image_size = (526, 957), shuffle = True, seed = 0, validation_split = 0.05, interpolation = 'bilinear', subset = 'training').map(lambda x : x/255)
lab_dataset_train = tf.keras.utils.image_dataset_from_directory(dir_lab, batch_size=5, labels = None, image_size = (526, 957), shuffle = True, seed = 0, validation_split = 0.05, subset = 'training', interpolation = 'nearest', color_mode = 'rgb')

img_dataset_validation = tf.keras.utils.image_dataset_from_directory(dir_img, batch_size=5, labels = None, image_size = (526, 957), shuffle = True, seed = 0, validation_split = 0.05, interpolation = 'bilinear', subset = 'validation').map(lambda x : x/255)
lab_dataset_validation = tf.keras.utils.image_dataset_from_directory(dir_lab, batch_size=5, labels = None, image_size = (526, 957), shuffle = True, seed = 0, validation_split = 0.05, subset = 'validation', interpolation = 'nearest', color_mode = 'rgb')

# Procesado 
img_ds_train = img_dataset_train.map(util.normalize)
img_ds_validation = img_dataset_validation.map(util.normalize)

lab_ds_train = lab_dataset_train.map(lambda x: util.map_fn(x, json.load(open('id_labels.json'))))
lab_ds_validation = lab_dataset_validation.map(lambda x: util.map_fn(x, json.load(open('id_labels.json'))))

dataset_train = tf.data.Dataset.zip((img_ds_train, lab_ds_train))
dataset_validation = tf.data.Dataset.zip((img_ds_validation, lab_ds_validation))

# Carga de modelo
segformer_model = models.TFSegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", num_labels = 20, ignore_mismatched_sizes=True
)

segformer_model.load_weights("../Resultados/modelos/gta/model_weights/meanIoU_loss")  # Carga de uno de nuestros modelos

segformer_model.summary()

# Definimos callback que guarda el modelo durante el entrenamiento
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='../Resultados/modelos/gta/model_weights/meanIoU_loss',
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

# Entrenamiento
segformer_model.compile(optimizer='adam', loss = util.mean_IoU_loss)

history = segformer_model.fit(dataset_train, validation_data = dataset_validation,
                    epochs= 10, batch_size=5, callbacks = [model_checkpoint_callback])

# Guardamos la evolución de la función de coste
np.save('../Resultados/modelos/gta/history/loss.npy', np.array(history.history['loss']))
np.save('../Resultados/modelos/gta/history/val_loss.npy', np.array(history.history['val_loss']))
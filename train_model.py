"""""
from tensorflow.io import read_file
from tensorflow.image import decode_image
import glob
import os
data_dir = './dataset/Dog/*.jpg'
for image in sorted(glob.glob(data_dir)):
    try:
        img = read_file(str(image))
        img = decode_image(img)
        if img.shape[2] != 3:
            print(image)
            os.remove(image)
    except Exception :
            os.remove(image)

exit()
"""""

# Importing required libraries
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Defining image size and batch size
img_height = 128
img_width = 128

batch_size = 32

# Creating training and testing sets
training_set = tf.keras.utils.image_dataset_from_directory('dataset',
                                                            validation_split=0.2,
                                                            subset="training",
                                                            seed=123,
                                                            image_size=(img_height, img_width),
                                                            batch_size=batch_size)

validation_set = tf.keras.utils.image_dataset_from_directory('dataset',
                                                             validation_split=0.2,
                                                             subset="validation",
                                                             seed=123,
                                                             image_size=(img_height, img_width),
                                                             batch_size=batch_size)
class_names = training_set.class_names
print(class_names)

num_classes = len(class_names)

# AUTOTUNE = tf.data.AUTOTUNE
#
# training_set = training_set.cache().shuffle(900).prefetch(buffer_size=AUTOTUNE)
# validation_set = validation_set.cache().prefetch(buffer_size=AUTOTUNE)

plt.figure(figsize=(10, 10))
for images, labels in training_set.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

data_augmentation = keras.Sequential(
  [
    keras.layers.RandomFlip("horizontal",
                      input_shape=(img_height, img_width,
                                  3)),
    keras.layers.RandomRotation(0.10),
    keras.layers.RandomZoom(0.10),
  ]
)


# Defining the model architecture
model = keras.Sequential([
    data_augmentation,
    keras.layers.Rescaling(1. / 255),
    keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(num_classes, name="outputs")
])

# Compiling the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

epochs = 10
runs = 3

for i in range(runs):
    print("run %d/%d"%(i +1, runs))
    history = model.fit(training_set,epochs=epochs,validation_data=validation_set)


# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# import tensorflow as tf
# import tensorflow_datasets as tfds
#
# #Correccion temporal (22/mayo/2022)
# #Tensorflow datasets tiene error al descargar el set de perros y gatos y lo solucionaron
# #el 16 de mayo pero sigue fallando en los colabs. Entonces se agrega esta linea adicional
# #Mas detalle aqui: https://github.com/tensorflow/datasets/issues/3918
# setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")
#
# #Descargar el set de datos de perros y gatos
# datos, metadatos = tfds.load('cats_vs_dogs', as_supervised=True, with_info=True)
#
# #Imprimir los metadatos para revisarlos
# metadatos
#
# #Una forma de mostrar 5 ejemplos del set
# tfds.as_dataframe(datos['train'].take(5), metadatos)
#
# #Otra forma de mostrar ejemplos del set
# tfds.show_examples(datos['train'], metadatos)
#
# #Manipular y visualizar el set
# #Lo pasamos a TAMANO_IMG (100x100) y a blanco y negro (solo para visualizar)
# import matplotlib.pyplot as plt
# import cv2
#
# plt.figure(figsize=(20,20))
#
# TAMANO_IMG=100
#
# for i, (imagen, etiqueta) in enumerate(datos['train'].take(25)):
#   imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
#   imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#   plt.subplot(5, 5, i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.imshow(imagen, cmap='gray')
#
# # Variable que contendra todos los pares de los datos (imagen y etiqueta) ya modificados (blanco y negro, 100x100)
# datos_entrenamiento = []
#
# for i, (imagen, etiqueta) in enumerate(datos['train']): #Todos los datos
#   imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
#   imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#   imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG, 1) #Cambiar tamano a 100,100,1
#   datos_entrenamiento.append([imagen, etiqueta])
#
# #Ver los datos del primer indice
# datos_entrenamiento[0]
#
# #Ver cuantos datos tengo en la variable
# len(datos_entrenamiento)
#
# #Preparar mis variables X (entradas) y y (etiquetas) separadas
#
# X = [] #imagenes de entrada (pixeles)
# y = [] #etiquetas (perro o gato)
#
# for imagen, etiqueta in datos_entrenamiento:
#   X.append(imagen)
#   y.append(etiqueta)
#
# X #####
#
# #Normalizar los datos de las X (imagenes). Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
# import numpy as np
#
# X = np.array(X).astype(float) / 255
#
# y ########
#
# #Convertir etiquetas en arreglo simple
# y = np.array(y)
#
# X.shape ##
#
# #Crear los modelos iniciales
# #Usan sigmoid como salida (en lugar de softmax) para mostrar como podria funcionar con dicha funcion de activacion.
# #Sigmoid regresa siempre datos entre 0 y 1. Realizamos el entrenamiento para al final considerar que si la respuesta se
# #acerca a 0, es un gato, y si se acerca a 1, es un perro.
#
# modeloDenso = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
#   tf.keras.layers.Dense(150, activation='relu'),
#   tf.keras.layers.Dense(150, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
#
# modeloCNN = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(100, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
#
# modeloCNN2 = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(250, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
#
#
# #Compilar modelos. Usar crossentropy binario ya que tenemos solo 2 opciones (perro o gato)
# modeloDenso.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# modeloCNN.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# modeloCNN2.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# from tensorflow import keras
# from keras.callbacks import TensorBoard
#
# #La variable de tensorboard se envia en el arreglo de "callbacks" (hay otros tipos de callbacks soportados)
# #En este caso guarda datos en la carpeta indicada en cada epoca, de manera que despues
# #Tensorboard los lee para hacer graficas
#
# tensorboardDenso = TensorBoard(log_dir='logs/denso')
# modeloDenso.fit(X, y, batch_size=32,
#                 validation_split=0.15,
#                 epochs=100,
#                 callbacks=[tensorboardDenso])
#
# # Convert the model.
# converter_denso = tf.lite.TFLiteConverter.from_keras_model(modeloDenso)
# tflite_model_denso = converter_denso.convert()
#
# # Save the model.
# with open('model_denso.tflite', 'wb') as f:
#   f.write(tflite_model_denso)
#
#
# tensorboardCNN = TensorBoard(log_dir='logs/cnn')
# modeloCNN.fit(X, y, batch_size=32,
#                 validation_split=0.15,
#                 epochs=100,
#                 callbacks=[tensorboardCNN])
#
# # Convert the model.
# converter_cnn = tf.lite.TFLiteConverter.from_keras_model(modeloCNN)
# tflite_model_cnn = converter_cnn.convert()
#
# # Save the model.
# with open('model_cnn.tflite', 'wb') as f:
#   f.write(tflite_model_cnn)
#
# tensorboardCNN2 = TensorBoard(log_dir='logs/cnn2')
# modeloCNN2.fit(X, y, batch_size=32,
#                 validation_split=0.15,
#                 epochs=100,
#                 callbacks=[tensorboardCNN2])
#
#
# # Convert the model.
# converter_cnn2 = tf.lite.TFLiteConverter.from_keras_model(modeloCNN2)
# tflite_model_cnn2 = converter_cnn2.convert()
#
# # Save the model.
# with open('model_cnn2.tflite', 'wb') as f:
#   f.write(tflite_model_cnn2)
#
#
# #ver las imagenes de la variable X sin modificaciones por aumento de datos
# plt.figure(figsize=(20, 8))
# for i in range(10):
#   plt.subplot(2, 5, i+1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.imshow(X[i].reshape(100, 100), cmap="gray")
#
# # Realizar el aumento de datos con varias transformaciones. Al final, graficar 10 como ejemplo
# from keras.preprocessing.image import ImageDataGenerator
#
# datagen = ImageDataGenerator(
#   rotation_range=30,
#   width_shift_range=0.2,
#   height_shift_range=0.2,
#   shear_range=15,
#   zoom_range=[0.7, 1.4],
#   horizontal_flip=True,
#   vertical_flip=True
# )
#
# datagen.fit(X)
#
# plt.figure(figsize=(20, 8))
#
# for imagen, etiqueta in datagen.flow(X, y, batch_size=10, shuffle=False):
#   for i in range(10):
#       plt.subplot(2, 5, i + 1)
#       plt.xticks([])
#       plt.yticks([])
#       plt.imshow(imagen[i].reshape(100, 100), cmap="gray")
#   break
#
# modeloDenso_AD = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(100, 100, 1)),
#   tf.keras.layers.Dense(150, activation='relu'),
#   tf.keras.layers.Dense(150, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# modeloCNN_AD = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(100, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# modeloCNN2_AD = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 1)),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#   tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
#   tf.keras.layers.MaxPooling2D(2, 2),
#
#   tf.keras.layers.Dropout(0.5),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(250, activation='relu'),
#   tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# modeloDenso_AD.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# modeloCNN_AD.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# modeloCNN2_AD.compile(optimizer='adam',
#                     loss='binary_crossentropy',
#                     metrics=['accuracy'])
#
# #Separar los datos de entrenamiento y los datos de pruebas en variables diferentes
#
# len(X) * .85 #19700
# len(X) - 19700 #3562
#
# X_entrenamiento = X[:19700]
# X_validacion = X[19700:]
#
# y_entrenamiento = y[:19700]
# y_validacion = y[19700:]
#
# #Usar la funcion flow del generador para crear un iterador que podamos enviar como entrenamiento a la funcion FIT del modelo
# data_gen_entrenamiento = datagen.flow(X_entrenamiento, y_entrenamiento, batch_size=32)
#
# tensorboardDenso_AD = TensorBoard(log_dir='logs/denso_AD')
#
# modeloDenso_AD.fit(
#     data_gen_entrenamiento,
#     epochs=100, batch_size=32,
#     validation_data=(X_validacion, y_validacion),
#     steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
#     validation_steps=int(np.ceil(len(X_validacion) / float(32))),
#     callbacks=[tensorboardDenso_AD]
# )
#
# tensorboardCNN_AD = TensorBoard(log_dir='logs-new/cnn_AD')
#
# modeloCNN_AD.fit(
#     data_gen_entrenamiento,
#     epochs=150, batch_size=32,
#     validation_data=(X_validacion, y_validacion),
#     steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
#     validation_steps=int(np.ceil(len(X_validacion) / float(32))),
#     callbacks=[tensorboardCNN_AD]
# )
#
# tensorboardCNN2_AD = TensorBoard(log_dir='logs/cnn2_AD')
#
# modeloCNN2_AD.fit(
#     data_gen_entrenamiento,
#     epochs=100, batch_size=32,
#     validation_data=(X_validacion, y_validacion),
#     steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
#     validation_steps=int(np.ceil(len(X_validacion) / float(32))),
#     callbacks=[tensorboardCNN2_AD]
# )
#
# modeloCNN_AD.save('perros-gatos-cnn-ad.h5')
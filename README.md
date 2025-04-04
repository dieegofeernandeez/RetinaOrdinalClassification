# Clasificación Ordinal de Imágenes de Retina

Este proyecto implementa un sistema de clasificación ordinal para imágenes de retina utilizando deep learning. Se enfoca en el diagnóstico de enfermedades oculares a partir de imágenes médicas, respetando la naturaleza ordinal de las etiquetas (por ejemplo, grados de severidad).

## Modelo

El modelo principal está basado en **ResNet50** con pesos preentrenados en ImageNet, al que se le añade un **bloque de atención espacial** para mejorar el enfoque en regiones relevantes de la imagen. La salida consiste en 4 logits que se interpretan como probabilidades acumulativas para clases ordinales (0 a 4).

## Estructura del Proyecto

- `model.py`: Definición del modelo con atención espacial y salida ordinal.
- `train.py`: Entrenamiento y validación del modelo usando *Ordinal Cross Entropy*.
- `data_loaders.py`: Carga y preprocesamiento de los datasets en formato `.npz`.
- `dataset.py`: Dataset personalizado con transformaciones.
- `preprocess.py`: Conversión de datasets originales (como MedMNIST o APTOS) a formato `.npz`.

## Entrenamiento

```bash
python train.py




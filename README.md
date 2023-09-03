# Transformers en segmentación de imágenes: rendimiento en el mundo real tras entrenamiento en imágenes sintéticas

Trabajo de fin de máster en el que se estudia la posibilidad de incluir imágenes sintéticas en el entrenamiento de un modelo de segmentación semántica basado en la arquitectura Transformer (modelo *Segformer*).

Autor: Lorenzo Pardo Chico

Turores: Valero Laparra Pérez-Muelas y Pablo Hernández Cámara

## Objetivos
El objetivo del trabajo es estudiar si el uso de imágenes sintéticas en el entrenamiento de un modelo de segmentación de imágenes mejora los resultados a la hora de aplicar el modelo a imágenes reales. De este modo, se ayudaría a solucionar uno de los principales inconvenientes a la hora de entrenar un modelo de segmentación que es la falta de imágenes etiquetadas.

Las imágenes artificiales están extraídas del videojuego GTA V, y se proponen seis modelos diferentes cuyos pesos entrenados proporcionamos en este repositorio:
1. Modelo entrenado al completo en Cityscapes
2. Modelo entrenado en Cityscapes y Fine Tuning del decoder en GTA V
3. Modelo entrenado en Cityscapes y Fine Tuning completo en GTA V
4. Modelo entrenado al completo en GTA V
5. Modelo entrenado en GTA V y Fine Tuning del decoder en Cityscapes
6.  Modelo entrenado en GTA V y Fine Tuning completo en Cityscapes

## Resultados
Para evaluar los modelos se han usado 500 imágenes aleatorias del dataset Mapillary y se han empleado 4 métricas: *meanIoU*, *Precision*, *Recall* y *F1-score*.    Si consideramos todas las categorías:

| Modelo          | meanIoU | Precision | Recall | F1-score |
|:---------------:|:-------:|:---------:|:------:|:--------:|
| City            | 0.75    | 0.82      | 0.89   | 0.85     |
| City + Dec GTA  | 0.70    | 0.80      | 0.85   | 0.82     |
| GTA + All City  | 0.65    | 0.81      | 0.76   | 0.79     |
| GTA + Dec City  | 0.62    | 0.75      | 0.78   | 0.77     |
| City + All GTA  | 0.58    | 0.73      | 0.75   | 0.74     |
| GTA             | 0.53    | 0.67      | 0.71   | 0.69     |

Para cada una de las categorías por seprado:

![](/Resultados/Comparacion_metricas.png "Resultados obtenidos las diferentes versiones del modelo Segformer MiT-B0")

Si mostramos la máscara de segmentación de unas cuantas imágenes para cada uno de los modelos:

![](/Resultados/img_comparison.png )

## Conclusión
En términos medios, los modelos que hacen uso de imágenes sintéticas en el entrenamiento no consiguen batir al modelo entrenado al completo sobre imágenes reales. Sin embargo, para alguna de las categorías sí que se observa una mejoría al introducir las imágenes artificiales. Por ejemplo, el modelo preentrenado en Cityscapes y reentrenado su decoder en las imágenes de GTA V consigue mejorar el IoU en las categorías *person* y *building*, que son de importancia vital en aplicaciones como la conducción autónoma. Si nos fijamos en las otras métricas, también observamos una mejoría en el rendimiento del modelo para ciertas categorías.


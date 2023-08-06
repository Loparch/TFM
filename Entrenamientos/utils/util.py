import tensorflow as tf

def normalize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  
    image = tf.image.per_image_standardization(tf.image.resize(image, size = [512, 512], method = 'bilinear'))  # Normalización por desviación estándar
    image = tf.transpose(image, perm = [0, 3, 1, 2])
    return image

def map_fn(image, label_to_color): #map from rgb to id
    
    image = tf.cast(image, tf.int32)
    # Crea un tensor de la misma forma que la imagen pero con valores de etiquetas
    label = tf.zeros_like(image[:,:,:,0], dtype=tf.int32)
    label_final = tf.zeros_like(image[:,:,:,0], dtype=tf.int32)

    # Itera sobre cada color en el diccionario y asigna la etiqueta correspondiente
    for color, label_value in label_to_color.items():
        mask = tf.reduce_all(tf.equal(image, color), axis=-1)
        label = tf.where(mask, x =tf.constant(label_value, dtype=tf.int32), y = 0)
        label_final = label_final + label

    return label_final

# loss function
def mean_IoU_loss(y_true, y_pred):
    label_interp_shape =  y_true.shape[1:]
    upsampled_logits = tf.image.resize(tf.transpose(y_pred, perm = [0, 2, 3, 1]) , size=label_interp_shape, method="bilinear")
    upsampled_logits_norm = tf.keras.layers.Softmax(axis = -1)(upsampled_logits)

    iou_loss = tf.zeros((1,))

    for c in range(upsampled_logits.shape[3]):
        y_true_bis = tf.cast((y_true == c), 'float')
        y_pred_bis = upsampled_logits_norm[:,:,:,c]
        numerator = tf.math.multiply(y_true_bis, y_pred_bis)
        denominator = y_true_bis + y_pred_bis - numerator
        iou = tf.math.reduce_sum(numerator)/tf.math.reduce_sum(denominator)
        iou_los = 1 - iou
        iou_loss = iou_loss + iou_los

    return iou_loss/upsampled_logits.shape[3]
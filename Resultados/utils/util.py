import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from PIL import Image


def get_labels(path, num_img):
  X = []

  for img in sorted(glob.glob(path + "/*"))[:num_img]:
    X_labels = np.array(Image.open(img), dtype=np.uint8)
    X_labels = tf.reshape(X_labels, (1, X_labels.shape[0], X_labels.shape[1]))
    X.append(tf.image.resize(tf.transpose(X_labels, perm = [1, 2, 0]), (526, 957), method = 'nearest'))
  
  X = np.array(X)
  X = np.reshape(X, (X.shape[:3]))

  return X

def map_colors(img_lab, color_dic):
    img_lab = np.array(img_lab)
    # Convierte la imagen a un tensor de tipo entero
    img_color = np.empty((img_lab.shape[0], img_lab.shape[1], 3))

    for i in range(img_color.shape[0]):
        for j in range(img_color.shape[1]):
            for label, color in color_dic.items():
                if img_lab[i,j] == int(label):
                    img_color[i,j,:] = np.array(tuple(color))

    return img_color/255

def map_fn(image, label_to_color): #map from rgb to id
    # Convierte la imagen a un tensor de tipo entero
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
    # upsample logits to the images' original size
    # `labels` is of shape (batch_size, height, width)
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


def show_image_and_mask(dataset, color_map = False, dic_colors = None):
    for image, mask in dataset.take(1):
        plt.figure(figsize = (15,15))
        plt.subplot(421)
        plt.imshow(image[0])
        plt.title("Image")
        plt.axis("off")
        plt.subplot(422)
        if color_map == False:
            plt.imshow(mask[0])
        else:
            plt.imshow(map_colors(mask[0], dic_colors))
        plt.title("Mask")
        plt.axis("off")
        plt.subplot(423)
        plt.imshow(image[1])
        plt.title("Image")
        plt.axis("off")
        plt.subplot(424)
        if color_map == False:
            plt.imshow(mask[1])
        else:
            plt.imshow(map_colors(mask[1], dic_colors))
        plt.title("Mask")
        plt.axis("off")
        plt.subplot(425)
        plt.imshow(image[2])
        plt.title("Image")
        plt.axis("off")
        plt.subplot(426)
        if color_map == False:
            plt.imshow(mask[2])
        else:
            plt.imshow(map_colors(mask[2], dic_colors))
        plt.title("Mask")
        plt.axis("off")
        plt.subplot(427)
        plt.imshow(image[3])
        plt.title("Image")
        plt.axis("off")
        plt.subplot(428)
        if color_map == False:
            plt.imshow(mask[3])
        else:
            plt.imshow(map_colors(mask[3], dic_colors))
        plt.title("Mask")
        plt.axis("off")

def normalize(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convierte a tipo float32
    image = tf.image.per_image_standardization(tf.image.resize(image, size = [512, 512], method = 'bilinear'))  # Normalización por desviación estándar
    image = tf.transpose(image, perm = [0, 3, 1, 2])
    return image


def map_categories(image, map):
    img = tf.cast(image, tf.int32)
    mask_final = tf.zeros_like(img)

    # Itera sobre cada color en el diccionario y asigna la etiqueta correspondiente
    for label_orig, label_final in map.items():
        mask = tf.math.equal(img, int(label_orig))
        label = tf.where(mask, x =tf.constant(label_final + 1, dtype=tf.int32), y = 0)
        mask_final = mask_final + label
    mask_0 = tf.math.equal(mask_final, 0)
    label_0 = tf.where(mask_0, x =tf.constant(20, dtype=tf.int32), y = 0)
    mask_final = tf.subtract(mask_final + label_0, 1)
    
    return mask_final


def post_process_output(output):
    output_ds = tf.data.Dataset.from_tensor_slices(output).batch(5)
    output_category = []
    for output in output_ds:
        upsampled_output = tf.image.resize(tf.transpose(output, perm = [0,2,3,1]), size = (526, 957), method = "bilinear")
        output_category.append(np.array(tf.math.argmax(upsampled_output, axis = 3)))

    output_categroy_array = np.vstack(output_category)
    output_category_ds = tf.data.Dataset.from_tensor_slices(output_categroy_array).batch(5)
        
    return output_category_ds


def calculate_metrics(results_ds,id_label = None, num_classes = 19, verbose = False):
    idlabel = {v: k for k, v in id_label.items()}
    iou_class = {k: 0 for k, v in idlabel.items()}
    precision_class = {k: 0 for k, v in idlabel.items()}
    recall_class = {k: 0 for k, v in idlabel.items()}
    F1_class = {k: 0 for k, v in idlabel.items()}

    if num_classes < 20:
        del iou_class['void'], precision_class['void'], recall_class['void'], F1_class['void']

    tp_labels = tf.keras.metrics.TruePositives()
    fp_labels = tf.keras.metrics.FalsePositives()
    fn_labels = tf.keras.metrics.FalseNegatives()

    for lab in range(len(iou_class.keys())):
        tp_labels.reset_state()
        fp_labels.reset_state()
        fn_labels.reset_state()
        for lab_truth, lab_pred in results_ds:
            pred_one_hot = tf.one_hot(lab_pred, num_classes)[:,:,:,lab]
            true_one_hot = tf.one_hot(lab_truth, num_classes)[:,:,:,lab]
            tp_labels.update_state(true_one_hot,pred_one_hot)
            fp_labels.update_state(true_one_hot,pred_one_hot)
            fn_labels.update_state(true_one_hot,pred_one_hot)
        
        iou = tp_labels.result().numpy()/(tp_labels.result().numpy() + fp_labels.result().numpy() + fn_labels.result().numpy())
        precision = tp_labels.result().numpy()/(tp_labels.result().numpy() + fp_labels.result().numpy())
        recall = tp_labels.result().numpy()/(tp_labels.result().numpy() + fn_labels.result().numpy())
        F1 = 2*precision*recall/(precision+recall)

        if verbose == True:
            print(f"IoU clase {id_label[str(lab)]}: ", iou)
            print(f"Precision clase {id_label[str(lab)]}: ", precision)
            print(f"Recall clase {id_label[str(lab)]}: ", recall)
            print(f"F1 clase {id_label[str(lab)]}: ", F1)
            
        iou_class[id_label[str(lab)]] = iou
        precision_class[id_label[str(lab)]] = precision
        recall_class[id_label[str(lab)]] = recall
        F1_class[id_label[str(lab)]] = F1

    iou_class = dict(sorted(iou_class.items(), key=lambda x: x[1], reverse=True))
    precision_class = dict(sorted(precision_class.items(), key=lambda x: x[1], reverse=True))
    recall_class = dict(sorted(recall_class.items(), key=lambda x: x[1], reverse=True))
    F1_class = dict(sorted(F1_class.items(), key=lambda x: x[1], reverse=True))
    

    tp_labels.reset_state()
    fp_labels.reset_state()
    fn_labels.reset_state()

    for lab_truth, lab_pred in results_ds:
        pred_one_hot = tf.one_hot(lab_pred, num_classes)
        true_one_hot = tf.one_hot(lab_truth, num_classes)
        tp_labels.update_state(true_one_hot,pred_one_hot)
        fp_labels.update_state(true_one_hot,pred_one_hot)
        fn_labels.update_state(true_one_hot,pred_one_hot)

    iou = tp_labels.result().numpy()/(tp_labels.result().numpy() + fp_labels.result().numpy() + fn_labels.result().numpy())
    precision = tp_labels.result().numpy()/(tp_labels.result().numpy() + fp_labels.result().numpy())
    recall = tp_labels.result().numpy()/(tp_labels.result().numpy() + fn_labels.result().numpy())
    F1 = 2*precision*recall/(precision+recall)

    if verbose == True:
        print("IoU media: ", iou)
        print("Precision media: ", precision)
        print("Recall media: ", recall)

    iou_class['Mean'] = iou
    precision_class['Mean'] = precision
    recall_class['Mean'] = recall
    F1_class['Mean'] = F1

    results = pd.DataFrame([iou_class, precision_class, recall_class, F1_class], index = ['IoU', 'Precision', 'Recall', 'F1']).T

    return results


def show_loss_history(loss, val_loss, file_name = 'loss_cityscapes_GTA_enocder_congelado', save = False):
    x = np.arange(1, len(loss) + 1)
    plt.figure()
    plt.plot(x, loss, color = 'black')
    plt.plot(x, val_loss, '--', color = 'black', linewidth=1)
    plt.grid(which = "both")
    plt.xlabel("Epoch", style = 'italic')
    plt.ylabel("$L_{meanIoU}$", style = 'italic')
    plt.xlim(left = 1, right = len(loss) + 1)
    plt.title('Función de coste')
    plt.legend(["Train", "Validation"], loc ="upper right")
    if save:
        plt.savefig(f"./Memoria/{file_name}.png", bbox_inches='tight', pad_inches = 2)

def map_colors_combined(img, lab, alpha = 0.75, dic_color = None):
    img = np.array(img)
    lab = np.array(lab)
    mask = np.zeros((lab.shape[0], lab.shape[1], 3))
    for i in range(lab.shape[0]):
        for j in range(lab.shape[1]):
            mask[i, j, :] = dic_color[str(lab[i,j])]
            
    mask_combined = img*(1-alpha) + mask*alpha/255

    return mask_combined

def show_results(img_ds, mask_truth_ds, predicted_mask_ds, size = (12,7), alpha = 0.9, file_name = "img_resultados_raw", dic_color = None, save = False):
    for img, mask_truth, predicted_mask in tf.data.Dataset.zip((img_ds, mask_truth_ds, predicted_mask_ds)).take(1):
        plt.figure(figsize = size)
        plt.subplot(511)
        plt.imshow(np.concatenate((img[0].numpy(), np.zeros((526,3,3)), map_colors_combined(img = img[0], lab = mask_truth[0],alpha = alpha, dic_color = dic_color), np.zeros((526,3,3)),map_colors_combined(img = img[0], lab = predicted_mask[0], alpha = alpha, dic_color = dic_color)), axis = 1))
        plt.axis("off")
        plt.title("               Imagen                     Máscara verdadera            Máscara predicha", loc= 'left')
        plt.subplot(512)
        plt.imshow(np.concatenate((img[1].numpy(), np.zeros((526,3,3)), map_colors_combined(img = img[1], lab = mask_truth[1], alpha = alpha, dic_color = dic_color), np.zeros((526,3,3)), map_colors_combined(img = img[1], lab = predicted_mask[1], alpha = alpha, dic_color = dic_color)), axis = 1))
        plt.axis("off")
        plt.subplot(513)
        plt.imshow(np.concatenate((img[2].numpy(),np.zeros((526,3,3)), map_colors_combined(img = img[2],lab = mask_truth[2], alpha = alpha, dic_color = dic_color), np.zeros((526,3,3)), map_colors_combined(img = img[2], lab = predicted_mask[2], alpha = alpha, dic_color = dic_color)), axis = 1))
        plt.axis("off")
        plt.subplot(514)
        plt.imshow(np.concatenate((img[3].numpy(),np.zeros((526,3,3)), map_colors_combined(img = img[3],  lab = mask_truth[3], alpha = alpha, dic_color = dic_color),np.zeros((526,3,3)), map_colors_combined(img = img[3], lab = predicted_mask[3], alpha = alpha, dic_color = dic_color)), axis = 1))
        plt.axis("off")
        plt.subplot(515)
        plt.imshow(np.concatenate((img[4].numpy(),np.zeros((526,3,3)), map_colors_combined(img = img[4],  lab = mask_truth[4], alpha = alpha, dic_color = dic_color),np.zeros((526,3,3)), map_colors_combined(img = img[4], lab = predicted_mask[4], alpha = alpha, dic_color = dic_color)), axis = 1))
        plt.axis("off")
        if save:
            plt.savefig(f"./Memoria/{file_name}.png", bbox_inches='tight', pad_inches = 2)


def show_results_comparison(img_ds, mask_truth_ds, predicted_mask_ds_1, predicted_mask_ds_2, predicted_mask_ds_3, predicted_mask_ds_4, predicted_mask_ds_5, predicted_mask_ds_6, size = (8,30), alpha = 0.9, file_name = "img_comparison", seed = 0,  dic_color = None, save = False):
    for img, mask_truth, pred_mask_1, pred_mask_2, pred_mask_3, pred_mask_4, pred_mask_5, pred_mask_6 in tf.data.Dataset.zip((img_ds, mask_truth_ds, predicted_mask_ds_1, predicted_mask_ds_2, predicted_mask_ds_3, predicted_mask_ds_4, predicted_mask_ds_5, predicted_mask_ds_6)).shuffle(10,seed = seed).take(1):
        fig = plt.figure(figsize = size)

        ax1 = fig.add_subplot(8,3,1)
        ax1.imshow(img[0].numpy())
        plt.title("Image", x = -0.5, y = 0.5, weight='bold')
        ax1.axis("off")
        ax2 = fig.add_subplot(8,3,2)
        ax2.imshow(img[1].numpy())
        ax2.axis("off")
        ax3 = fig.add_subplot(8,3,3)
        ax3.imshow(img[2].numpy())
        ax3.axis("off")

        ax4 = fig.add_subplot(8,3,4)
        ax4.imshow(map_colors_combined(img = img[0], lab = pred_mask_1[0], alpha = alpha, dic_color = dic_color))
        plt.title("City", x = -0.5, y = 0.5, weight='bold')
        ax4.axis("off")
        ax5 = fig.add_subplot(8,3,5)
        ax5.imshow(map_colors_combined(img = img[1], lab = pred_mask_1[1], alpha = alpha, dic_color = dic_color))
        ax5.axis("off")
        ax6 = fig.add_subplot(8,3,6)
        ax6.imshow(map_colors_combined(img = img[2], lab = pred_mask_1[2], alpha = alpha, dic_color = dic_color))
        ax6.axis("off")

        ax7 = fig.add_subplot(8,3,7)
        ax7.imshow(map_colors_combined(img = img[0], lab = pred_mask_2[0], alpha = alpha, dic_color = dic_color))
        plt.title("City + Dec GTA", x = -0.5, y = 0.5, weight='bold')
        ax7.axis("off")
        ax8 = fig.add_subplot(8,3,8)
        ax8.imshow(map_colors_combined(img = img[1], lab = pred_mask_2[1], alpha = alpha, dic_color = dic_color))
        ax8.axis("off")
        ax9 = fig.add_subplot(8,3,9)
        ax9.imshow(map_colors_combined(img = img[2], lab = pred_mask_2[2], alpha = alpha, dic_color = dic_color))
        ax9.axis("off")

        ax10 = fig.add_subplot(8,3,10)
        ax10.imshow(map_colors_combined(img = img[0], lab = pred_mask_3[0], alpha = alpha, dic_color = dic_color))
        plt.title("City + All GTA", x = -0.5, y = 0.5, weight='bold')
        ax10.axis("off")
        ax11 = fig.add_subplot(8,3,11)
        ax11.imshow(map_colors_combined(img = img[1], lab = pred_mask_3[1], alpha = alpha, dic_color = dic_color))
        ax11.axis("off")
        ax12 = fig.add_subplot(8,3,12)
        ax12.imshow(map_colors_combined(img = img[2], lab = pred_mask_3[2], alpha = alpha, dic_color = dic_color))
        ax12.axis("off")

        ax13 = fig.add_subplot(8,3,13)
        ax13.imshow(map_colors_combined(img = img[0], lab = pred_mask_4[0], alpha = alpha, dic_color = dic_color))
        plt.title("GTA", x = -0.5, y = 0.5, weight='bold')
        ax13.axis("off")
        ax14 = fig.add_subplot(8,3,14)
        ax14.imshow(map_colors_combined(img = img[1], lab = pred_mask_4[1], alpha = alpha, dic_color = dic_color))
        ax14.axis("off")
        ax15 = fig.add_subplot(8,3,15)
        ax15.imshow(map_colors_combined(img = img[2], lab = pred_mask_4[2], alpha = alpha, dic_color = dic_color))
        ax15.axis("off")

        ax16 = fig.add_subplot(8,3,16)
        ax16.imshow(map_colors_combined(img = img[0], lab = pred_mask_5[0], alpha = alpha, dic_color = dic_color))
        plt.title("GTA + Dec City", x = -0.5, y = 0.5, weight='bold')
        ax16.axis("off")
        ax17 = fig.add_subplot(8,3,17)
        ax17.imshow(map_colors_combined(img = img[1], lab = pred_mask_5[1], alpha = alpha, dic_color = dic_color))
        ax17.axis("off")
        ax18 = fig.add_subplot(8,3,18)
        ax18.imshow(map_colors_combined(img = img[2], lab = pred_mask_5[2], alpha = alpha, dic_color = dic_color))
        ax18.axis("off")

        ax19 = fig.add_subplot(8,3,19)
        ax19.imshow(map_colors_combined(img = img[0], lab = pred_mask_6[0], alpha = alpha, dic_color = dic_color))
        plt.title("GTA + All City", x = -0.5, y = 0.5, weight='bold')
        ax19.axis("off")
        ax20 = fig.add_subplot(8,3,20)
        ax20.imshow(map_colors_combined(img = img[1], lab = pred_mask_6[1], alpha = alpha, dic_color = dic_color))
        ax20.axis("off")
        ax21 = fig.add_subplot(8,3,21)
        ax21.imshow(map_colors_combined(img = img[2], lab = pred_mask_6[2], alpha = alpha, dic_color = dic_color))
        ax21.axis("off")

        ax22 = fig.add_subplot(8,3,22)
        ax22.imshow(map_colors_combined(img = img[0], lab = mask_truth[0], alpha = alpha, dic_color = dic_color))
        plt.title("Ground Truth", x = -0.5, y = 0.5, weight='bold')
        ax22.axis("off")
        ax23 = fig.add_subplot(8,3,23)
        ax23.imshow(map_colors_combined(img = img[1], lab = mask_truth[1], alpha = alpha, dic_color = dic_color))
        ax23.axis("off")
        ax24 = fig.add_subplot(8,3,24)
        ax24.imshow(map_colors_combined(img = img[2], lab = mask_truth[2], alpha = alpha, dic_color = dic_color))
        ax24.axis("off")
        if save:
            plt.savefig(f"./Memoria/{file_name}.png", bbox_inches='tight', pad_inches = 2)





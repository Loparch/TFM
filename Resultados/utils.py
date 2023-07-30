import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
from PIL import Image

from huggingface_hub import hf_hub_url, cached_download
import json


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

    return iou_class, precision_class, recall_class, F1_class


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
        plt.title("Imagen   Máscara verdadera   Máscara predicha")
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





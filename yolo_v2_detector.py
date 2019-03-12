# A YOLO-V2 network performing object detection. Ported to Keras(YAD2K), pretrained on COCO dataset.
# It will load the model with the pretrained weights, print its layers,
# try to detect objects on the given image and save the image with the predicted
# boxes in the "/output/" path.
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from PIL import Image, ImageDraw, ImageFont
import imghdr

IMAGE_NAME = 'maradona.jpg' #Image to infer

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    """Filters YOLO boxes by thresholding on object and class confidence.
    
    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    
    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes
    
    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold. 
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """
    
    # Compute box scores
    box_scores = box_confidence*box_class_probs;
    
    # Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    
    # Create a filtering mask based on "box_class_scores" by using "threshold".
    filtering_mask = box_class_scores >= threshold
    

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    return scores, boxes, classes

def iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = max(box1[1],box2[1])
    yi1 = max(box1[0],box2[0])
    xi2 = min(box1[3],box2[3])
    yi2 = min(box1[2],box2[2])
    inter_area = max(yi2-yi1,0)*max(xi2-xi1,0)

    # Calculate the Union area by using Formula
    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    union_area =box1_area+box2_area-inter_area
    
    # compute the IoU
    iou = inter_area/union_area    
    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    
    Arguments:
    scores -- tensor of shape (None,), output of yolo_filter_boxes()
    boxes -- tensor of shape (None, 4), output of yolo_filter_boxes() that have been scaled to the image size (see later)
    classes -- tensor of shape (None,), output of yolo_filter_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box
    
    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) # initialize variable max_boxes_tensor
    
    # get the list of indices corresponding to boxes we keep
    nms_indices = tf.image.non_max_suppression(
    boxes,
    scores,
    max_boxes_tensor,
    iou_threshold)
   
    # select only nms_indices from scores, boxes and classes
    scores = K.gather(scores,nms_indices)
    boxes =  K.gather(boxes,nms_indices)
    classes =K.gather(classes,nms_indices)
    
    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (a lot of boxes) to your predicted boxes along with their scores, box coordinates and classes.
    
    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)
    image_shape -- tensor of shape (2,) containing the input shape, in this notebook we use (608., 608.) (has to be float32 dtype)
    max_boxes -- integer, maximum number of predicted boxes you'd like
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
    
    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """
    
    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[:]

    # Convert boxes to be ready for filtering functions 
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # perform Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes

def preprocess_image(img_path, model_image_size):
    image = Image.open(img_path)
    size = image.size
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data, size

# Input image (608, 608, 3)
# The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
# After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
# Each cell in a 19x19 grid over the input image gives 425 numbers.
# 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes.
# 85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and and 80 is the number of classes we'd like to detect
# We then select only few boxes based on:
# Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
# Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
# This gives us YOLO's final output.


# Run the graph on an image
def load_model_and_infer(sess, image_file):
    """
    Load the model as a graph in sesssion and Run it
    to predict boxes for "image_file". Prints and plots the preditions.
    
    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image_file -- name of an image stored in the "images" folder.
    
    Returns:
    out_scores -- tensor of shape (None, ), scores of the predicted boxes
    out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
    out_classes -- tensor of shape (None, ), class index of the predicted boxes
    
    Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes. 
    """
    # Load and preprocess image for inference
    image, image_data, image_size = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # Define classes, anchors and image shape.
    class_names = read_classes("model/coco_classes.txt")
    anchors = read_anchors("model/yolo_anchors.txt")
    image_shape = (float(image_size[1]),float(image_size[0]))    

    # Load the pretrained model.
    yolo_model = load_model("model/yolo.h5")

    # print a summary of the layer's model.
    yolo_model.summary()

    # Convert output of the model to bounding box tensors.
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

    # Filter boxes.
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
    
    # Infer predictions on the image.
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})

    # Print predictions info
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("output", image_file), quality=90)
    # Display the results in the notebook
    output_image = scipy.misc.imread(os.path.join("output", image_file))
    imshow(output_image)
    
    return out_scores, out_boxes, out_classes

sess = K.get_session()
out_scores, out_boxes, out_classes = load_model_and_infer(sess, IMAGE_NAME)

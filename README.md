# CNN-YOLO-V2-Object-Detection
A YOLO-V2 network performing object detection. Ported to Keras(YAD2K [[4]](https://github.com/allanzelener/YAD2K)), pretrained on COCO dataset.

Uses a YOLO-V2 setup transfered to Keras from DarkNet, uses pretrained weights [[5]](https://pjreddie.com/darknet/yolo/)    on COCO Dataset.
The Input image is resized to an (608, 608, 3) tensor, goes through the CNN resulting in a 
(19,19,5,85) dimensional output. After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
Each cell in a 19x19 grid over the input image gives 425 numbers.
 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes.
85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw) has 5 numbers, and and 80 is the number of classes we'd like to detect

We then select only few boxes based on:
* Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
* Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes

**This gives us YOLO's final output. Bounding boxes with estimated probabillity for a certain class of the COCO dataset.**
### Example output
<p float="center">
  <img src="/output/test.jpg" alt="drawing" width="800"/>
  <b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</b>
  <img src="/output/football.jpg" alt="drawing" width="800"/>
</p> 

*I do not hold the rights of the presented images. [[6]](https://tryolabs.com/images/blog/post-images/2018-03-01-guide-to-visual-question-answering/visual-question-answering.aa6ecaa1.jpg),[[7]](https://previews.123rf.com/images/larash/larash1803/larash180300013/110497762-rome-italy-june-17-2014-police-car-horse-cart-with-a-coach-for-tourists-on-the-streets-of-rome-italy.jpg)* 

--------------------------------------------------------------------------------
## Usage

Execute `yolo_v2_detector` python script. It will load the model with the pretrained weights, print its layers,
try to detect objects on the given image and save the image with the predicted boxes in the `"/output/"` path. If you want to run the detector on a custom image you can change the `image_file` parameter with your image name and store it to `"/images/"` path.

## Generate Model for Keras (YAD2K)

- Clone the [YAD2K Library](https://github.com/allanzelener/YAD2K) to your PC
- Open terminal from the cloned directory
- Download model cfg and weights for YOLOv2-608x608 (COCO) from the [YOLO website](https://pjreddie.com/darknet/yolov2/). 
- Copy and paste the downloaded weights and cfg files to the YAD2K master directory
- Run `python yad2k.py yolov2.cfg yolov2.weights model_data/yolo.h5` on the terminal and the h5 file will be generated.
- Move the generated h5 file to `model` folder

*Based on code and lectures from the Deeplearning.ai specialization.*


## References

Deeplearning.ai CNN course on Coursera.[[1]](https://www.coursera.org/learn/convolutional-neural-networks)  
Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi. You Only Look Once: Unified, Real-Time Object Detection.[[2]](https://arxiv.org/abs/1506.02640)  
Joseph Redmon, Ali Farhadi. YOLO9000: Better, Faster, Stronger[[3]](https://arxiv.org/abs/1612.08242)  
Allan Zelener, YAD2K: Yet Another Darknet 2 Keras.[[4]](https://github.com/allanzelener/YAD2K)   
The official YOLO website.[[5]](https://pjreddie.com/darknet/yolo/)  
Images used source.[[6]](https://tryolabs.com/images/blog/post-images/2018-03-01-guide-to-visual-question-answering/visual-question-answering.aa6ecaa1.jpg),[[7]](https://previews.123rf.com/images/larash/larash1803/larash180300013/110497762-rome-italy-june-17-2014-police-car-horse-cart-with-a-coach-for-tourists-on-the-streets-of-rome-italy.jpg)

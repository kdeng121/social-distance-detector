# Social Distancing Detector

Social distancing detector using [OpenCV](https://opencv.org/) computer vision library and [YOLO](https://pjreddie.com/darknet/yolo/) object detection library.


## Usage

```python .\main.py --video true --video_path '.\PATH\TO\VIDEO.mp4'```: Social distancing detection from video in real-time

```python .\main.py --video true --video_path '.\PATH\TO\VIDEO.mp4' --output true```: Social distancing detection from video with output video 

```python .\main.py --image true --image_path '.\PATH\TO\IMAGE.jpg'```: Social distancing detection image 


Please download yolov3.weights from YOLO and add it to your repo (file is too large to add to Git): [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

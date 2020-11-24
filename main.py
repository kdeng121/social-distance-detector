import cv2
import numpy as np 
import argparse
import time
import imutils
from scipy.spatial import distance as dist

parser = argparse.ArgumentParser()
parser.add_argument('--video', help="True/False", default=False)
parser.add_argument('--image', help="True/False", default=False)
parser.add_argument('--video_path', help="Path of video file")
parser.add_argument('--image_path', help="Path of image to detect objects")
parser.add_argument('--verbose', help="To print statements", default=True)
parser.add_argument('--output', help="True/False", default=False)
args = parser.parse_args()

"""
CONSTANTS
"""
MIN_CONF = 0.3 # Minimum probabaility of detected class to filter out weak detections
NMS_THRESH = 0.3 # Threshold used for non-maxima suppression (removing redundant bounding boxes)
MIN_DISTANCE = 175 # Minimum safe distance in pixels that two people can be from each other
PERSON_ID = 0 # Id for Person class
WRITER = None # Writer to output video

"""
Loads the YOLO object detection library with configuration files
"""
def load_yolo():
    net = cv2.dnn.readNet("yolov3-416.weights", "yolov3-416.cfg")
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

"""
Given a file path to an image, load it using OpenCV
"""
def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net, outputLayers):			
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    centroids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if class_id == PERSON_ID and conf > MIN_CONF:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
                centroids.append((center_x, center_y))
    return boxes, confs, class_ids, centroids
            
def draw_labels(boxes, confs, centroids, colors, class_ids, classes, img): 
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    actualBoxesDetected = 0

    # Final calculation variables
    final_boxes = []
    final_confs = []
    final_centroids = []
    
    for i in range(len(boxes)):
        if i in indexes:
            actualBoxesDetected+=1
            final_boxes.append(boxes[i])
            final_confs.append(confs[i])
            final_centroids.append(centroids[i])
    
    violations = social_distance_violations(final_centroids)
    
    for i in range(len(final_boxes)):
        x, y, w, h = final_boxes[i]
        # label = str(classes[class_ids[i]])
        center_x, center_y = final_centroids[i]
        # color = colors[i]
        color = (255, 255, 255)

        # Draw red box if object has violated social distancing
        if (i in violations):
            color = (0, 0, 255)

        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        cv2.putText(img, 'person', (x, y - 5), font, 1, color, 1)
        cv2.circle(img, (center_x, center_y), 5, color)

        text = "Social Distancing Violations: {}".format(len(violations))
        cv2.putText(img, text, (10, img.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    # Write to an output video file
    global WRITER
    output = args.output
    if (output and WRITER is None):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        WRITER = cv2.VideoWriter('output.avi', fourcc, 8,
            (img.shape[1], img.shape[0]), True)  

    if (WRITER is not None):
        WRITER.write(img)
    else:             
        cv2.imshow("Image", img)

"""
Given array of centroids, return set of indexes that violated social distance threshold
"""    
def social_distance_violations(centroids):
    violate = set()

    if (len(centroids) < 2):
        return violate

    D = dist.cdist(centroids, centroids, metric="euclidean")

    for i in range(0, D.shape[0]):
        for j in range(i + 1, D.shape[1]):
            # Check distance between all pairs and compare to the minimum distance threshold (in pixels)
            if D[i, j] < MIN_DISTANCE:
                violate.add(i)
                violate.add(j)
    return violate

def image_detect(img_path): 
    model, classes, colors, output_layers = load_yolo()
    image, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids, centroids = get_box_dimensions(outputs, height, width)
    draw_labels(boxes, confs, centroids, colors, class_ids, classes, image)
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

def video_detect(video_path):
    model, classes, colors, output_layers = load_yolo()
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()

        if (frame is None):
            break

        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs, class_ids, centroids = get_box_dimensions(outputs, height, width)
        draw_labels(boxes, confs, centroids, colors, class_ids, classes, frame)
            
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()

if __name__ == '__main__':
    video_play = args.video
    image = args.image

    if video_play:
        video_path = args.video_path
        if args.verbose:
            print('Opening '+video_path+" ... ")
            if args.output:
                print('Writing to output file... this may take some time...')
            else:
                print('Press q to quit')
        video_detect(video_path)
        print('DONE!')

    if image:
        image_path = args.image_path
        if args.verbose:
            print("Opening "+image_path+" .... ")
        image_detect(image_path)
    
    cv2.destroyAllWindows()
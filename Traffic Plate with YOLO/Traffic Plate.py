"""
eo sensor: scans the surroundings with 360 cameras.
data : https://universe.roboflow.com/university-km5u7/traffic-sign-detection-yolov8-awuus
"""
# import libraries

from ultralytics import YOLO 

model = YOLO("yolov8n.pt")

model.train(
    data = "data/data.yaml", # data
    epochs = 5, # number of training epochs
    imgsz = 640, # image size
    batch = 16, # mini batch size adjusted depending on hardware
    name = "data", # output folder name
    lr0 = 0.01, # initial learning rate
    optimizer = "SGD", # alternative ADAM
    weight_decay = 0.0005, # weight penalty prevents overfitting
    momentum = 0.935, # SGD momentum
    patience = 50, # patience duration for early stopping
    workers = 2, # data loader worker count
    device = "cuda", # cpu or cuda
    save = True, # save models
    save_period = 1, # saving period
    val = True, # validation values at the end of each epoch
    verbose = True # to show what's happening in terminal
)

"""
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
            1/2         0G     0.7178      3.356       1.14         37        640: 100% ━━━━━━━━━━━━ 147/147 0.5it/s 4:43
box_loss = 0.1-0.3 range is sufficient
cls_loss = should not go below 1
dfl_loss = should be between 0.5-1

Images = Images
Instance = Detected Objects
Box(P) = Accuracy of detected boxes
R = How many of the real objects were captured
mAP50 = Box detection success (for 70% and above)
mAP50-95 = In more difficult thresholds (60% and above is successful)
-------------------------------------------------------------------------------------------------------------
Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 7/7 0.8it/s 8.9s
all        219        260      0.686      0.669      0.734      0.567
hump        22         22      0.558      0.745      0.665      0.471
no entry    30         34      0.447      0.548      0.502      0.401
no overt    21         21      0.529      0.476      0.532      0.468  # overtaking
no stopping 30         30      0.383        0.7      0.668       0.54
no u turn   20         21      0.406      0.762      0.692      0.544
parking     21         32      0.916      0.682      0.847      0.646
roadwork    22         26      0.835      0.584      0.809      0.512
roundabout  22         23      0.927      0.565       0.78      0.548
speedl40    29         30      0.908      0.657      0.863      0.711 # speed limit 40
stop        20         21      0.953      0.969      0.985      0.832

"""


# Finished.
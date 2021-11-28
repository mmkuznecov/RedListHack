import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

def find_tiger(img, detector):
    height, width, channels = img.shape
    
    outputs = detector.predict(img)

    v = Visualizer(img,
                   metadata=detector.metadata, 
                   scale=1,
                   instance_mode=ColorMode.IMAGE   
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
   
    count = len(list(outputs["instances"].pred_boxes))
    return v.get_image(), count

if __name__ == '__main__':
    i = cv2.imread("001_214.jpg")
    i = find_tiger(i)
    cv2.imwrite("tiger.jpg", i)
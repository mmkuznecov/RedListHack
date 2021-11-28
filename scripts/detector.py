import detectron2

import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetCatalog, MetadataCatalog

import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools

def get_animals_dicts(metadata, data_path):
    dataset_dicts = []
    for i, (label, night, path, bboxes, height, width) in enumerate(metadata.iloc):
        record = {}

        filename = os.path.join(data_path, path)
        record['image_id'] = i
        record['file_name'] = filename
        record['height'] = height
        record['width'] = width
        
        if type(bboxes) != float:
            bboxes = bboxes.split('|')
            for i in range(len(bboxes)):
                bboxes[i] = list(map(lambda x: max(0, float(x)), bboxes[i].split()))
        else:
            bboxes = []

        objs = []
        for bbox in bboxes:
            obj = {
                'bbox': bbox,
                    'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': label,
                'iscrowd': 0
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def get_animals_dicts_test(metadata, data_path):
    dataset_dicts = []
    for i, (label, night, path, height, width) in enumerate(metadata.iloc):
        record = {}
        filename = os.path.join(data_path, path)
        record['image_id'] = i
        record['file_name'] = filename
        record['height'] = height
        record['width'] = width
        dataset_dicts.append(record)
    return dataset_dicts

def init_catalog(catalog_name, metadata, path, classes, train=True):
    get_dict = get_animals_dicts if train else get_animals_dicts_test
    DatasetCatalog.register(catalog_name, lambda: get_dict(metadata, path))
    MetadataCatalog.get(catalog_name).set(thing_classes=classes)
    
class Detector:
    def __init__(self, model_config, train_catalog, learning_rate,
                 max_iter, bs_per_image, n_classes, threshold,
                 n_workers=1,model_name='last'):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config)) 
        self.cfg.DATALOADER.NUM_WORKERS = n_workers
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)
        self.cfg.DATASETS.TRAIN = (train_catalog,)
        self.cfg.DATASETS.TEST = (train_catalog,)
        self.cfg.SOLVER.IMS_PER_BATCH = n_workers
        self.cfg.SOLVER.BASE_LR = learning_rate
        self.cfg.SOLVER.MAX_ITER = max_iter  
        self.cfg.TEST.EVAL_PERIOD = max_iter + 1
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = bs_per_image  
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
        self.threshold = threshold
        self.cfg.OUTPUT_DIR = os.path.join(self.cfg.OUTPUT_DIR, model_name)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.metadata = MetadataCatalog.get(train_catalog)
        self.predictor = None
        
    def set_threshold(self, threshold):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        if self.predictor is not None:
            self.predictor = DefaultPredictor(self.cfg)
            
    def load_weights(self, weights):
        self.cfg.MODEL.WEIGHTS = weights
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.predictor = DefaultPredictor(self.cfg)

    def train(self):
        self.trainer = DefaultTrainer(self.cfg) 
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.predictor = DefaultPredictor(self.cfg)
        
    def predict(self, img):
        if self.predictor is None:
            self.predictor = DefaultPredictor(self.cfg)
        outputs = self.predictor(img)
        return outputs
    
    def predict_class(self, img):
        outputs = self.predictor(img)
        classes = outputs['instances'].pred_classes
        scores = outputs['instances'].scores
        if len(classes) > 0:
            idx = np.arange(len(classes)).tolist()
            idx.sort(key=lambda x: scores[x])
            return classes[idx[-1]], scores[idx[-1]]
        else:
            return None, None
        
    def predict_bbox(self, img):
        outputs = self.predictor(img)
        bbox = outputs['instances'].pred_boxes
        scores = outputs['instances'].scores
        if len(bbox) > 0:
            idx = np.arange(len(bbox)).tolist()
            idx.sort(key=lambda x: scores[x])
            return bbox[idx[-1]], scores[idx[-1]]
        else:
            return None, None
    
    def predict_class_bbox(self, img):
        outputs = self.predictor(img)
        classes = outputs['instances'].pred_classes
        bbox = outputs['instances'].pred_boxes
        scores = outputs['instances'].scores
        if len(bbox) > 0:
            idx = np.arange(len(bbox)).tolist()
            idx.sort(key=lambda x: scores[x])
            return classes[idx[-1]], bbox[idx[-1]], scores[idx[-1]]
        else:
            return None, None
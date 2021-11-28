#!/usr/local/bin/python3
# coding: utf8
from fastapi import FastAPI
from fastapi import File, UploadFile
import nest_asyncio
from pydantic import BaseModel
from tiger_rect import find_tiger

from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import uvicorn
import numpy as np
import io
import shutil
import pandas as pd

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import base64

import requests
import cv2

import os, os.path
import asyncio
import time
from settings import dirs

from clean_file_names import clean_file_name, clean_file_names

from detector import get_animals_dicts, get_animals_dicts_test, init_catalog
from detector import Detector

model_config =  dirs['ModelConfig']
learning_rate = dirs['LearningRate']
max_iter = dirs['MaxIter']
bs_per_image = dirs['BsPerImage']
classes = dirs['Classes']
threshold = dirs['Threshold']
model_name=dirs['ModelName']
metadata_path=dirs['MetadataPath']
ash_threshold=dirs['AshThreshold']

detector = Detector(model_config, 'animals_train_full', learning_rate,
                    max_iter, bs_per_image, len(classes), threshold,
                    model_name=model_name)

weights_path = os.path.join(detector.cfg.OUTPUT_DIR, 'model_final.pth')

#print(detector.cfg.OUTPUT_DIR)
#print(weights_path)
if os.path.isfile(weights_path):
    detector.load_weights(weights_path)
    print('Model weights loaded')


def sort_buf():
    print('1')
    path = dirs['Buffer']
    for fname in os.listdir(path):
        fname_ = clean_file_name(fname)
        if fname_ != fname:
            src = os.path.join(path, fname)
            dst = os.path.join(path, fname)
            copyfile(src, dst)
            os.remove(src)

    d = {'id':[], 'label':[], 'night':[], 'path':[], 'bbox':[], 'height':[], 'width':[]}
    idx_ = 0
    for fname in os.listdir(path):
        print('2')
        fpath = os.path.join(path, fname)
        img = cv2.imread(fpath)
        outputs = detector.predict(img)
        scores = outputs['instances'].scores
        labels = outputs['instances'].pred_classes
        bboxes = outputs['instances'].pred_boxes
        idx = np.arange(len(scores))
        idx = list(filter(lambda x: scores[x] > ash_threshold, idx))
        bboxes, labels = bboxes[idx], labels[idx]
        bboxes = bboxes.tensor.cpu().numpy().tolist()
        labels = labels.tensor.cpu().numpy().tolist()
        if len(labels) > 0:
            print('3')
            d['id'].append(idx_)
            d['label'].append(labels[0])
            d['night'].append(None)
            d['path'].append(fpath)
            height, width = img.shape[:2]
            d['height'].append(height)
            d['width'].append(width)
            bboxes = [' '.join(map(str, bbox)) for bbox in bboxes]
            bboxes = '|'.join(bboxes)
            d['bbox'].append(bboxes)
            shutil.move(fpath, os.path.join(dirs[classes[labels[0]]], fname))
            idx_ += 1
        else:
            print('4')
            shutil.move(fpath, os.path.join(dirs['Uncertain'], fname))
    df = pd.DataFrame(d)
    df.set_index('id', inplace=True)
    metadata = pd.read_csv(metadata_path, index_col='id')
    metadata = pd.concat([metadata, df])
    metadata.to_csv(metadata_path)

async def train_model(path):
    #make some train shit
    pass


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def main():
    return "This is simple API to interact tiger recognition system. Send recognition requests on /api/upload"

@app.get("/api/check")
def check():
    unsorted = len(os.listdir(dirs["Buffer"]))
    return {"unsorted_images" : unsorted }

@app.get("/api/sort")
async def sort():
    sort_buf()
    time.sleep(10)
    sorts = {dir_:len(os.listdir(dirs[dir_])) for dir_ in list(dirs.keys())[:2 + len(classes)]}
    return sorts

@app.get("/api/train")
async def train():
    train_model(dirs)
    time.sleep(10)
    return {"new_metric" : 1.46}

@app.post('/api/upload')
async def find(file: UploadFile = File(...)):
    #img_data = requests.get(req.link).content
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img, count = find_tiger(img, detector)

    res, im_png = cv2.imencode(".png", img)
    headers = {"count" : str(count)}
    return {"img":base64.b64encode(im_png.tobytes())}
    #return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png", headers = headers)
    #return {"result":push(req.text, req.lenght, req.temperature)}

@app.post('/api/upload_alt')
async def find_alt(file: UploadFile = File(...)):
    #img_data = requests.get(req.link).content
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img, count = find_tiger(img, detector)

    res, im_png = cv2.imencode(".png", img)
    headers = {"count" : str(count)}
    #return {"img":base64.b64encode(im_png.tobytes())}
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png", headers = headers)
    #return {"result":push(req.text, req.lenght, req.temperature)}

nest_asyncio.apply()
uvicorn.run(app, host="0.0.0.0", port=5000,timeout_keep_alive=10000)
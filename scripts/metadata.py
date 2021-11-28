import pandas as pd
import os
import configparser
from clean_file_names import clean_file_name, clean_file_names
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_train_metadata(data_path, metadata_path, classes, detector=None):
    d = {'id':[], 'label':[], 'night':[], 'path':[], 'bbox':[], 'height':[], 'width':[]}
    idx = 0
    for class_id, class_name in enumerate(classes):
        if class_name != classes[0]:
            objects_df = pd.read_csv(os.path.join(data_path, class_name, 'objects.csv'))
            objects_df['id'] = objects_df['id'].apply(clean_file_name)
            objects_df = objects_df.set_index('id')
        for fname in tqdm(os.listdir(os.path.join(data_path, class_name, 'images'))):
            d['id'].append(idx)
            d['label'].append(class_id)
            d['night'].append(None)
            d['path'].append(os.path.join(class_name, 'images', fname))
            img = plt.imread(os.path.join(data_path, class_name, 'images', fname))
            height, width = img.shape[:2]
            d['height'].append(height)
            d['width'].append(width)
            if class_name != classes[0] and fname in objects_df.index:
                bboxes = objects_df.loc[fname]['bbox']
                bboxes = [bboxes] if type(bboxes) == str else bboxes
                bboxes = '|'.join(bboxes)
                d['bbox'].append(bboxes)
            elif detector is not None:
                bboxes, _ = detector.predict_bbox(img)
                if bboxes is not None:
                    bboxes = bboxes.tensor.cpu().tolist()
                    bboxes = '|'.join(map(lambda x: ' '.join(map(lambda y: str(y), x)), bboxes))
                d['bbox'].append(bboxes)
            else:
                d['bbox'].append(None)
            idx += 1
    df = pd.DataFrame(d)
    df.set_index('id', inplace=True)
    df.to_csv(os.path.join(metadata_path, 'metadata_train.csv'))
    
def create_test_metadata(data_path, metadata_path, classes):
    d = {'id':[], 'label':[], 'night':[], 'path':[], 'height':[], 'width':[]}
    idx = 0
    for class_id, class_name in enumerate(classes):
        for fname in tqdm(os.listdir(os.path.join(data_path, class_name, 'images'))):
            d['id'].append(idx)
            d['label'].append(class_id)
            d['night'].append(None)
            d['path'].append(os.path.join(class_name, 'images', fname))
            height, width = plt.imread(os.path.join(data_path, class_name, 'images', fname)).shape[:2]
            d['height'].append(height)
            d['width'].append(width)
            idx += 1
    df = pd.DataFrame(d)
    df.set_index('id', inplace=True)
    df.to_csv(os.path.join(metadata_path, 'metadata_test.csv'))
    
def create_metadata(detector=None):
    config = configparser.ConfigParser()
    config.read('settings.ini');
    train_path = config['Path']['train_path']
    test_path = config['Path']['test_path']
    metadata_path = config['Path']['metadata_path']
    other_class = config['Class']['other']
    classes  = [other_class] + config['Class']['classes'].split()
    
    metadata_train_path = os.path.join(metadata_path, 'metadata_train.csv')
    metadata_test_path = os.path.join(metadata_path, 'metadata_test.csv')
    
    clean_file_names([train_path, test_path], classes)
    
    create_train_metadata(train_path, metadata_path, classes, detector)
    create_test_metadata(test_path, metadata_path, classes)
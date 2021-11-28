from detector import Detector
import os
import configparser
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('settings.ini');
    input_path = config['Path']['to_crop_path']
    output_path = config['Path']['cropped_path']
    other_class = config['Class']['other']
    classes  = [other_class] + config['Class']['classes'].split()
    threshold = float(config['Training']['threshold'])
    model_config = config['Training']['model_config']
    best_model_name = config['Models']['best_model_name']

    detector = Detector(model_config, 'placeholder', 0.1, 1, 1, len(classes),
                        threshold, model_name=best_model_name)
    detector.load_weights(os.path.join(detector.cfg.OUTPUT_DIR, 'model_final.pth'))

    for fname in tqdm(os.listdir(input_path)):    
        img = plt.imread(os.path.join(input_path, fname))
        bbox, _ = detector.predict_bbox(img)
        bbox = bbox.tensor.cpu().numpy()
        if len(bbox) != 1:
            print(fname)
        else:
            bbox = list(map(int, bbox[0]))
            img_cropped = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            plt.imsave(os.path.join(output_path, fname), img_cropped)
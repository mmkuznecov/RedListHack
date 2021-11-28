from detector import Detector
import os
import configparser
import pandas as pd
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('settings.ini');
    solutions_path  = config['Path']['solutions_path']
    final_test_path  = config['Path']['final_test_path']
    other_class = config['Class']['other']
    classes  = [other_class] + config['Class']['classes'].split()
    threshold = float(config['Training']['threshold'])
    model_config = config['Training']['model_config']
    best_model_name = config['Models']['best_model_name']

    class2id = {'tiger':1, 'leopard':2, 'other':3}

    detector = Detector(model_config, 'placeholder', 0.1, 1, 1, len(classes),
                        threshold, model_name=best_model_name)
    detector.load_weights(os.path.join(detector.cfg.OUTPUT_DIR, 'model_final.pth'))

    fnames = []
    labels = []
    for fname in tqdm(os.listdir(final_test_path)):    
        img = cv2.imread(os.path.join(final_test_path, fname))
        fnames.append(fname)
        label, _ = detector.predict_class(img)
        label = 3 if label == None else class2id[classes[label]]
        labels.append(label)

    solution = pd.DataFrame([fnames, labels], index=['id', 'class']).T
    solution.to_csv(os.path.join(solutions_path, 'labels.csv'), index=False)
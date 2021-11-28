from shutil import copyfile
from transliterate import translit
import re 
import os

def has_cyrillic(text):
    return bool(re.search('[а-яА-Я]', text))

def clean_file_name(fname):
    if ' ' in fname or '—' in fname or has_cyrillic(fname):
        fname = fname.replace(' ', '_')
        fname = fname.replace('—', '_')
        fname = translit(fname, language_code='ru', reversed=True)
    return fname

def clean_file_names(data_paths, classes):
    for data_path in data_paths:
        for class_id, class_name in enumerate(classes):
            for fname in os.listdir(os.path.join(data_path, class_name, 'images')):
                fname_ = clean_file_name(fname)
                if fname_ != fname:
                    src = os.path.join(data_path, class_name, 'images', fname)
                    dst = os.path.join(data_path, class_name, 'images', fname)
                    copyfile(src, dst)
                    os.remove(src)
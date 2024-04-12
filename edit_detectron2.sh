#!/bin/bash

# Пути к файлам
FILE_PATH_TRANSFORM="/miniconda/envs/py38/lib/python3.8/site-packages/detectron2/data/transforms/transform.py"
FILE_PATH_PASCAL_VOC="/miniconda/envs/py38/lib/python3.8/site-packages/detectron2/data/datasets/pascal_voc.py"
FILE_PATH_PASCAL_VOC_EVAL="/miniconda/envs/py38/lib/python3.8/site-packages/detectron2/evaluation/pascal_voc_evaluation.py"

# Изменение файла transform.py
if [ -f "$FILE_PATH_TRANSFORM" ]; then
    sed -i 's/def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):/def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0):/' "$FILE_PATH_TRANSFORM"
    echo "Файл $FILE_PATH_TRANSFORM успешно изменен."
else
    echo "Ошибка: Файл $FILE_PATH_TRANSFORM не найден."
fi

# Изменение файла pascal_voc.py
if [ -f "$FILE_PATH_PASCAL_VOC" ]; then
    sed -i 's/fileids = np.loadtxt(f, dtype=np.str)/fileids = np.loadtxt(f, dtype=str)/' "$FILE_PATH_PASCAL_VOC"
    echo "Файл $FILE_PATH_PASCAL_VOC успешно изменен."
else
    echo "Ошибка: Файл $FILE_PATH_PASCAL_VOC не найден."
fi

# Изменение файла pascal_voc_evaluation.py
if [ -f "$FILE_PATH_PASCAL_VOC_EVAL" ]; then
    sed -i 's/difficult = np.array(\[x\["difficult"\] for x in R\]).astype(np.bool)/difficult = np.array(\[x\["difficult"\] for x in R\]).astype(bool)/' "$FILE_PATH_PASCAL_VOC_EVAL"
    echo "Файл $FILE_PATH_PASCAL_VOC_EVAL успешно изменен."
else
    echo "Ошибка: Файл $FILE_PATH_PASCAL_VOC_EVAL не найден."
fi

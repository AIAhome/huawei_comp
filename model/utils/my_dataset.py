# coding=utf-8
import json
import os
from pathlib import Path
import datasets
from typing import List,Tuple,Dict
import string
from PIL import Image
import numpy as np
import re


# import torch
# from detectron2.data.transforms import ResizeTransform, TransformList
logger = datasets.logging.get_logger(__name__)
_CITATION = """\
author=Yangzhe Peng
"""
_DESCRIPTION = """\
CSIG2022中英文小票识别数据集加载
"""

DATA_DIR='my_dataset/train'

_URL = 'model/train.tgz'

def load_image(image_path):
    image = Image.open(image_path)
    w, h = image.size
    return image, (w, h)


class MyDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for MyDataset"""
    def __init__(self,**kwargs):
        """BuilderConfig for MyDataset.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MyDatasetConfig, self).__init__(**kwargs)

class MyDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MyDatasetConfig(name='en' ,version=datasets.Version("1.0.0"), description='the English recipt'),
        MyDatasetConfig(name='zh' ,version=datasets.Version("1.0.0"), description='the Chinese recipt'),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "words": datasets.Sequence(datasets.Value("string")),
                "bboxes": datasets.Sequence(datasets.Sequence(datasets.Sequence(datasets.Value("int64")))),
                # "image": datasets.Array3D(shape=(3, myProcessor.resized_size[0], myProcessor.resized_size[1]), dtype="uint8"),
                "image_path": datasets.Value("string"),
                "gt": datasets.Features({
                    'date': datasets.Value('string'),
                    'total': datasets.Value('string'),
                    'company': datasets.Value('string'),
                    'tax': datasets.Value('string'),
                    'items': datasets.Sequence(
                        datasets.Features({
                            'name': datasets.Value('string'),
                            'price': datasets.Value('string'),
                            'cnt': datasets.Value('string') 
                            })
                        )
                    })
                # "labels": datasets.Sequence(
                #         datasets.ClassLabel(
                #             names=label2id
                #         )
                #     )
                }),
            supervised_keys=None,
            citation=_CITATION,
            homepage="https://competition.huaweicloud.com/information/1000041696/introduction",
        )

    @property
    def manual_download_instructions(self):
        return (
            '''
            to use this dataset, please download it from the competition homepage manually,
            and use datasets.load_dataset('my_dataset.py','en'/'zh',data_dir='path/to/folder')
            '''
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        # downloaded_file = dl_manager.download_and_extract(_URL)
        # dest = Path(downloaded_file)
        dest = Path(dl_manager.manual_dir)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,gen_kwargs={'filepath': dest/'train'}
            ),            
        ]

    def _generate_examples(self,filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ocr_dir = os.path.join(filepath, "ocr_labels")
        img_dir = os.path.join(filepath, "images")
        gt_dir = os.path.join(filepath,'ie_labels')
        lang_files = []
        for fname in os.listdir(img_dir):
            if self.config.name == 'zh':
                if 'zh' in fname:
                    lang_files.append(fname)
            else:
                if 'en' in fname:
                    lang_files.append(fname)

        for guid, fname in enumerate(sorted(lang_files)):
            name, ext = os.path.splitext(fname)
            ocr_file_path = os.path.join(ocr_dir, name + ".json")
            with open(ocr_file_path, "r", encoding="utf8") as f:
                ocr = json.load(f)

            gt_file_path = os.path.join(gt_dir , name+'.json')
            with open(gt_file_path,'r',encoding='utf-8') as f:
                gt = json.load(f)
            
            image_file_path = os.path.join(img_dir, fname)
            
            boxes = [item['points'] for item in ocr]
            words = [item['label'] for item in ocr]

            yield guid, {"id": str(guid), "words": words, "image_path": image_file_path, "bboxes": boxes, "gt": gt}
            # return format like this
            # {'date': '2022-03-29',
            #  'total': '$67.00',
            #  'company': 'DOLLAR TREE',
            #  'tax': '',
            #  'items': {'name': ['drag suits', 'microkini'],
            #   'price': ['$21.00', '$25.00'],
            #   'cnt': ['2.00', '1.00']}}

            # image,words,normalized_box,word_labels = myProcessor.process_example({'image_path':image_file_path,'words':words,'bboxes':boxes,'gt':gt})
            # yield guid, {"id": str(guid), "words": words, "image": image, "bboxes": normalized_box, "labels": word_labels}

from transformers import AutoProcessor
# import datasets
from typing import List,Tuple,Dict
import string
from PIL import Image
import numpy as np
import re


Entities_list = [
    "company",
    "date",
    "total",
    "tax",
    "name",
    "cnt",
    "price"
]
id2label = {}#FIXME: not consider the [PAD](-100), so there might be some error when using
label2id = {}
id2label[0] = 'O'
label2id['O'] = 0
id=1
for label in Entities_list:
    id2label[id] = 'I-%s'%label
    id2label[id+1] = 'B-%s'%label
    label2id['I-%s'%label] = id
    label2id['B-%s'%label] = id + 1
    id += 2


### helper func ####
def process_money(money):
    '''
    处理货币符号的位置和保留两位小数
    '''
    if money == '':
        return money
    if '$' in money:
        num = money.replace('$','')
        num = '%.2f' % float(num)
        money = '$' + num
    elif '￥' in money:
        num = money.replace('￥','')
        num = '%.2f' % float(num)
        money = '￥' + num
    else:
        money = '%.2f' % float(money)
    return money
def text2iob_label(words: List[str], entities: Dict) -> List[List[str]]:
    '''
    get iob label corresponding each word
    '''
    result_tags = ['O'] * len(words)

    # flatten the ie_labels
    exactly_entities_label = []
    for key in ['company', 'date', 'total', 'tax']:
        exactly_entities_label.append([key, entities[key]])
    # for item in entities['items']:
    #     for key in item:
    #         exactly_entities_label.append([key, item[key]])
    for key in entities['items']:
        for item in entities['items'][key]:
            exactly_entities_label.append([key, item])
    for entity_type, entity_value in exactly_entities_label:
        flag = 0
        if entity_type=='date':
            date = set(entity_value.split('-'))
            pattern = re.compile(r'\d+')
        for idx,word in enumerate(words):
            if entity_type == 'date':
                num = set(pattern.findall(word))
                if not num.isdisjoint(date):
                    if result_tags[idx]=='O' :
                        if flag == 0 :
                            result_tags[idx] = 'B-%s' % entity_type
                            flag = 1
                        elif result_tags[idx-1]=='B-date':
                            result_tags[idx] = 'I-%s' % entity_type      
            elif entity_type == 'company' or entity_type == 'name':
                if (word in entity_value or entity_value in word ) and result_tags[idx]=='O' :
                    if flag == 0  :
                        result_tags[idx] = 'B-%s' % entity_type
                        flag = 1
                    else:
                        result_tags[idx] = 'I-%s' % entity_type
            else: # 剩下的都是money类字段，需要精确匹配
                try:
                    if '*' in word:
                        word = word.split('*')[0]# 有时候会写成单价*数量的形式
                    elif '元' in word:
                        word = word.replace('元','')# 有时候会加一个‘元’字
                    if words[idx - 1] == '$':#有的时候货币符号会跟数字分开
                        word = '$'+word
                    elif words[idx -1] == '￥':
                        word = '￥'+word
                    money = process_money(word)
                except:
                    continue
                if money == entity_value and result_tags[idx]=='O' :
                    result_tags[idx] = 'B-%s' % entity_type
                    break
    rtn_tags = []
    for tag in result_tags:
        rtn_tags.append(label2id[tag])
    return rtn_tags

def normalize_bbox(bbox, size):
    return [
        [int(1000 * bbox[0][0] / size[0]),#left top
        int(1000 * bbox[0][1] / size[1]),
        # [int(1000 * bbox[1][0] / size[0]),# right top
        # int(1000 * bbox[1][1] / size[1])],
        int(1000 * bbox[2][0] / size[0]),# right down
        int(1000 * bbox[2][1] / size[1])],
        # [int(1000 * bbox[3][0] / size[0]),# left down
        # int(1000 * bbox[3][1] / size[1])],
    ]
######### helper func end ##############

class MyDataProcessor():
    def __init__(self,resized_size = (480,960),train_en=True):
        self.resized_size = resized_size
        if train_en:
            self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)  
        else:
            self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base-chinese", apply_ocr=False)  

    def process_example(self,examples):
        # images_path = examples['image_path']
        # words = examples['words']
        # boxes = examples['bboxes']
        # gt = examples['gt']

        images = []
        wordsList = []
        boxesList = []
        labelList = []
        for i in range(len(examples['id'])):
            image_path = examples['image_path'][i]
            words = examples['words'][i]
            boxes = examples['bboxes'][i]
            gt = examples['gt'][i]

            image = Image.open(image_path).convert("RGB")

            word_labels = text2iob_label(words,gt)

            # resize image
            
            width = image.width
            height = image.height
            image = image.resize(self.resized_size,Image.BILINEAR)
            x_scale = self.resized_size[0] / width
            y_scale = self.resized_size[1] / height

            normalized_boxes = []
            for box in boxes:
                # get resized images's boxes coordinate
                resized_box = [ [point[0]*x_scale , point[1]*y_scale] for point in box]
                normalized_boxes += normalize_bbox(resized_box,self.resized_size)
            
            images.append(image)
            wordsList.append(words)
            boxesList.append(normalized_boxes)
            labelList.append(word_labels)
            
        # return image,words,normalized_boxes,word_labels

        # images = []
        # words = []
        # boxes = []
        # word_labels = []
        # for example in examples:
        #     image,words,boxes,word_labels = prepare_one_example(example)
        #     images.append(image)
        #     words.append(words)
        #     boxes.append(boxes)
        #     word_labels.append(word_labels)
        encoding = self.processor(images, wordsList, boxes=boxesList, word_labels=labelList, truncation=True, padding="max_length")

        return encoding


if __name__=='__main__':
    from datasets import load_dataset 
    en_dataset = load_dataset("my_dataset.py",'en',data_dir='data')
    # zh_dataset = load_dataset('my_dataset.py','zh',data_dir='data')
    print(en_dataset)
    # print(zh_dataset)

    processor = MyDataProcessor()

    from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D

    # we need to define custom features for `set_format` (used later on) to work properly
    features = Features({
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),# the LayoutLMv3 processor output pixel_values to be (3,224,224) instead of (3,resized_size[0],resized_size[1])
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'attention_mask': Sequence(Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),# [168, 103, 550, 136]
        'labels': Sequence(feature=Value(dtype='int64')),# [-100, 1, -100, 2, 2, 0, 0, 0, -100, -100, -100, 0, 0, -100, ...]
    })

    en_dataset = en_dataset["train"].map(
        processor.process_example,
        batched=True,
        remove_columns=en_dataset["train"].column_names,
        features=features,
        batch_size=50
    )
    example = en_dataset[0]
    processor.processor.tokenizer.decode(example["input_ids"])
    en_dataset.set_format("torch")
    for id, label in zip(en_dataset[0]["input_ids"], en_dataset[0]["labels"]):
        print(processor.processor.tokenizer.decode([id]), label.item())
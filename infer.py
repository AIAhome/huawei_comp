import cv2
from pathlib import Path
import glob
from model.customize_service import BaselineService
import json
import os

def read_dataset(dataset_path:str):
    image_files_path = Path(dataset_path+'/images').as_posix()
    image_files = sorted(glob.glob(image_files_path+'/*.*'))

    ocr_files_path = Path(dataset_path+'/ocr_labels').as_posix()
    ocr_files = sorted(glob.glob(ocr_files_path+'/*.*'))

    dataset = []
    for id,(img,ocr) in enumerate(zip(image_files,ocr_files)):
        image = cv2.imread(image_files[id])

        with open(ocr_files[id], 'r', encoding='utf-8') as f:
            ocr = json.load(f)
        
        data = {}
        data['id'] = id #实际似乎是文件名称，但因为用不上这个，就不做特殊处理了
        data['image'] = image
        data['ocr'] = ocr
        dataset.append(data)
    return dataset


if __name__=='__main__':
    #flexible, change when use
    dataset_path = 'data/train'
    model_path = 'model/saved/'
    eval_idx = [i for i in range(60,80)]+[j for j in range(2460,2480)]# english and chinese 



    import pickle
    # dataset = read_dataset(dataset_path)
    # print('finish reading dataset')

    # pickle.dump(dataset,open('dataset.pkl','wb'))

    dataset = pickle.load(open('dataset.pkl','rb'))    
    infer_api=BaselineService("", model_path)
    
    final_result = []
    for i in eval_idx:
        data = dataset[i]
        request_data={
            "file1":{
                "test_data.pkl": data
            }
        }

        input_data=infer_api._preprocess(request_data)
        result=infer_api._inference(input_data)
        final_result.append(infer_api._postprocess(result))

        # print(final_result)
    filename = 'res_v3.json'
    if os.path.exists(filename):
        os.remove(filename)
    json.dump(final_result, open(filename, 'wt'), indent=4, ensure_ascii=False)

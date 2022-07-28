from transformers import AutoModelForTokenClassification
from transformers import AutoProcessor
from PIL import Image
from typing import Dict, List
import os
import torch
import pickle
import numpy as np
import re
from copy import deepcopy

# self implement
from model.utils.dataProcessor import normalize_bbox

from model_service.pytorch_model_service import PTServingBaseService#BUG: change this before push

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def is_chinese_data(data):
    if isinstance(data, dict):
        data = list(data.values())
    if isinstance(data, list):
        return sum([is_chinese_data(d) for d in data]) != 0
    return is_Chinese(data)
              
# # 自己新建一个类，名字随便起，继承于PTServingBaseService类
class BaselineService(PTServingBaseService):# BUG: change when push

#     def __init__(self, model_name, model_path):
#         super(BaselineService, self).__init__(model_name, model_path)

# class BaselineService():
    def __init__(self, model_name, model_path) -> None:

        # 这句可以获得你当前模型包里model这个文件夹的路径，这个dir_path就这么写，这句不用改动
        dir_path = os.path.dirname(os.path.realpath(model_path))
        # 这句可以根据dir_path来找到你想要的任何model目录下的文件，这里我们定义了模型文件的路径
        ckpt_path = os.path.join(dir_path, 'saved')


        # 后面需要你初始化你的模型权重加载，配置一些全局变量等等，自由发挥
        self.en_model = AutoModelForTokenClassification.from_pretrained(ckpt_path+'/en/checkpoint-6340')
        self.zh_model = AutoModelForTokenClassification.from_pretrained(ckpt_path+'/zh/checkpoint-2248')
        # self.en_processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)  
        # self.zh_processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)  
        self.processor = AutoProcessor.from_pretrained(ckpt_path+'/en/checkpoint-6340', apply_ocr=False)  
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'


    # 下面三个函数，推理判分时，默认会依次按照 _preprocess -> _inference -> _postprocess 的顺序执行
    def _preprocess(self, request_data):
        # ‼️这里不用管ModelArts官方文档怎么说，这里在比赛中，默认一次只有一个数据读入，并且读入数据是pickle格式，以下几行循环内容完全不用改动
        data = None
        for k, v in request_data.items():
            for file_name, file_content in v.items():
                data = pickle.load(file_content) # use this when push code to huawei #BUG: change when push
                # data = file_content #use this in my own computer

        # 图片数据，为ndarray格式
        image = data['image']

        # ocr数据，字典格式，包含"label"和"points"两个键：
        #   - label: 格式为string
        #   - points: 格式为[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]，数据为float类型
        ocr = data['ocr']      
        boxes = []
        words = []
        for item in ocr:
            boxes.append(item['points'])
            words.append(item['label'])

        # 数据id，选手一般无需使用
        id = data['id']

        isChinese = is_chinese_data(words)

        image = Image.fromarray(image).convert("RGB")

        # resize image
        width = image.width
        height = image.height
        image = image.resize((480,960),Image.BILINEAR)#FIXME: i hard code the resized_size
        x_scale = 480 / width
        y_scale = 960 / height

        normalized_boxes = []
        for box in boxes:
            # get resized images's boxes coordinate
            resized_box = [ [point[0]*x_scale , point[1]*y_scale] for point in box]
            normalized_boxes += normalize_bbox(resized_box,(480,960))

        # if isChinese:
        #     encoding = self.zh_processor(image, words, boxes=normalized_boxes, return_tensors="pt")
        # else:
        #     encoding = self.en_processor(image, words, boxes=normalized_boxes, return_tensors="pt")
        encoding = self.processor(image, words, boxes=normalized_boxes, return_tensors="pt")

        return {'isChinese':isChinese,'encoding':encoding,'words':words}

    def iob2entity(self,input_ids:List[int],iob_labels:List[str]) -> Dict :
        result = dict(
            company="",
            date="",
            total="",
            tax="",
            items=[]
        )
        for key in result.keys():
            if key == 'items':
                continue
            value = ''
            idx = []
            for i,label in enumerate(iob_labels):
                if label[2:] == key:
                    idx.append(i)
                else:
                    try:
                        if iob_labels[i-1][2:] == key and iob_labels[i+1][2:] == key:#有的地方可能会对中间的一个token预测出错（很常见）
                            idx.append(i)
                    except IndexError:
                        pass
            value = self.processor.tokenizer.decode(input_ids[idx])
            result[key] = value

        # 找到所有的的name，cnt，price
        values = {}
        for key in ['name','cnt','price']:
            values[key]=[]
            flag = 0
            for i,label in enumerate(iob_labels):
                if label[2:] == key and flag == 0 :#找到单词的开头
                    value_idx = [i]
                    flag = 1
                elif label[2:] == key and flag == 1:# 找到单词的中间部分
                    value_idx.append(i)
                elif label[2:] != key and flag == 1:# 单词后面的那个token
                    flag = 0
                    value = self.processor.tokenizer.decode(input_ids[value_idx])
                    values[key].append(value)

        item_cnt = max(len(values['name']),len(values['cnt']),len(values['price']))
        for i in range(item_cnt):
            new_item = {'name':'','cnt':'','price':''}
            for key in new_item.keys():
                try:
                    new_item[key] = values[key][i]
                except:
                    pass
            result['items'].append(new_item)
        return result

    # 用于推理
    def _inference(self, input_data):
        if input_data['isChinese']:
            model = self.zh_model
        else: 
            model = self.en_model
        model = model.to(self.device)
        encoding = input_data['encoding'].to(self.device)

        model.eval()
        with torch.no_grad():
            output = model(**encoding)
            logits = output.logits
            predictions = logits.argmax(-1).squeeze().tolist()
            true_predictions = [self.en_model.config.id2label[pred] for pred in predictions] # there is no difference between en and zh in id2label
        
        return self.iob2entity(input_data['encoding']['input_ids'][0],true_predictions),input_data['words']


    def _postprocess(self, data):
        # return data[0]
        data, words = data
        scripts = words
        def guess_date_from_script(script):
            pattern = re.compile(r'\d+')
            def possible_date(s: str):
                def next_template(y_year, t_month, t_day):
                    if y_year['Option'] == 1:
                        return 'year'
                    if t_month['Option'] == 1:
                        return 'month'
                    if t_day['Option'] == 1:
                        return 'day'
                    return None

                dict_template = {'Fixed': False, 'NumberIndex': [], 'Option': 0}
                all_numbers = pattern.findall(s)[:3]

                if len(all_numbers) < 3:
                    return [None, None, None], False

                year_template   = deepcopy(dict_template)
                month_template  = deepcopy(dict_template)
                day_template    = deepcopy(dict_template)
                templates = [year_template, month_template, day_template]

                for i, number in enumerate(all_numbers):
                    if len(number) == 4 and int(number) >= 1990 and int(number) <= 2022:
                        year_template['Fixed'] = True
                        year_template['NumberIndex'] = [i]
                        year_template['Option'] = 1
                        continue
                    if len(number) >= 3:
                        continue
                    if int(number) >= 0 and int(number) <= 22 and year_template['Fixed'] == False:
                        year_template["NumberIndex"].append(i)
                        year_template['Option'] += 1
                    if int(number) >= 1 and int(number) <= 12:
                        month_template["NumberIndex"].append(i)
                        month_template['Option'] += 1
                    if int(number) >= 1 and int(number) <= 31:
                        day_template["NumberIndex"].append(i)
                        day_template['Option'] += 1

                result_date = [None, None, None]

                for i in range(3):
                    t_name = next_template(year_template, month_template, day_template)
                    if t_name is None:
                        break
                    targetIndex = {'year': 0, 'month': 1, 'day': 2}[t_name]

                    template = templates[targetIndex]
                    numberIndex = template["NumberIndex"][0]
                    result_date[targetIndex] = all_numbers[numberIndex]

                    for t in templates:
                        for j, n in enumerate(t["NumberIndex"]):
                            if n == numberIndex:
                                t["Option"] -= 1
                                t["NumberIndex"] = t["NumberIndex"][:j] + t["NumberIndex"][j+1:]
                                break

                def format_date(date):
                    if date[0] is None or date[1] is None or date[2] is None:
                        return date
                    if len(date[0]) == 2:
                        date[0] = "20" + date[0]
                    if len(date[1]) == 1:
                        date[1] = "0" + date[1]
                    if len(date[2]) == 1:
                        date[2] = "0" + date[2]
                    return date
                result_date = format_date(result_date)
                return result_date, False if None in result_date else True

            res = [possible_date(s) for s in script]
            res_ok = [r[1] for r in res]
            res_date = [r[0] for r in res]
            if sum(res_ok) != 1:
                return None
            return '-'.join(res_date[sum([r*i for i,r in enumerate(res_ok)])])



        def is_Chinese(word):
            for ch in word:
                if '\u4e00' <= ch <= '\u9fff':
                    return True
            return False
        def is_chinese_data(data):
            if isinstance(data, dict):
                data = list(data.values())
            if isinstance(data, list):
                return sum([is_chinese_data(d) for d in data]) != 0
            return is_Chinese(data)

        #判断左右匹配符号位置
        def bracket(word, left, right):
            start, end = 0, -1
            b_l, b_r = word.find(left), word.find(right)

            if b_l < b_r:
                index = (b_l, b_r)
            else:
                index = (start, end)
            return index

        def stringQ2B(ustring):
            """把字符串全角转半角"""
            def Q2B(uchar):
                """单个字符 全角转半角"""
                inside_code = ord(uchar)
                if inside_code == 0x3000:
                    inside_code = 0x0020
                else:
                    inside_code -= 0xfee0
                if inside_code < 0x0020 or inside_code > 0x7e: #转完之后不是半角字符返回原来的字符
                    return uchar
                return chr(inside_code)
            return "".join([Q2B(uchar) for uchar in ustring])

        def process_money(money):
            '''
            处理货币符号的位置和保留两位小数
            '''
            money = stringQ2B(money)
            if money == '':
                return money
            money = money.replace('S','$')
            money = money.replace('¥','￥')
            money = money.replace('O','0')
            money = money.split(' ')[0]# 有时候一个money字段里混了好几个数字
            if '$' in money:
                num = money.replace('$','')
                if num == '':
                    money = ''
                    return money
                num = num.replace('*','')
                num = num.replace('-','')
                num = '%.2f' % float(num)
                money = '$' + num
            elif '￥' in money:
                num = money.replace('￥','')
                if num == '':
                    money = ''
                    return money
                num = '%.2f' % float(num)
                money = '￥' + num
            else:
                try: 
                    money = '%.2f' % float(money)
                except:
                    pass
            return money

        ChineseData = is_chinese_data(data)
        def parse_date(is_chinese: bool, date: str) -> str:
            def format_date(l_date):
                if l_date[0] is None or l_date[1] is None or l_date[2] is None:
                    return l_date
                if len(l_date[0]) == 2:
                    l_date[0] = "20" + l_date[0]
                if len(l_date[1]) == 1:
                    l_date[1] = "0" + l_date[1]
                if len(l_date[2]) == 1:
                    l_date[2] = "0" + l_date[2]
                return l_date
            def numbers2str(l_date):
                return '-'.join(l_date)
            def next_template(y_year, t_month, t_day):
                if y_year['Option'] == 1:
                    return 'year'
                if t_month['Option'] == 1:
                    return 'month'
                if t_day['Option'] == 1:
                    return 'day'
                return None

            pattern = re.compile(r'\d+')
            all_numbers = pattern.findall(date)[:3]
            if len(all_numbers) < 3:
                return '2022-03-13' if is_chinese else '2022-03-29'

            if is_chinese:
                all_numbers = format_date(all_numbers)
                return numbers2str(all_numbers)

            dict_template = {'Fixed': False, 'NumberIndex': [], 'Option': 0}
            year_template   = deepcopy(dict_template)
            month_template  = deepcopy(dict_template)
            day_template    = deepcopy(dict_template)
            templates = [year_template, month_template, day_template]

            for i, number in enumerate(all_numbers):
                if len(number) == 4 and int(number) >= 1990 and int(number) <= 2022:
                    year_template['Fixed'] = True
                    year_template['NumberIndex'] = [i]
                    year_template['Option'] = 1
                    continue
                if len(number) >= 3:
                    continue
                if int(number) >= 0 and int(number) <= 22 and year_template['Fixed'] == False:
                    year_template["NumberIndex"].append(i)
                    year_template['Option'] += 1
                if int(number) >= 1 and int(number) <= 12:
                    month_template["NumberIndex"].append(i)
                    month_template['Option'] += 1
                if int(number) >= 1 and int(number) <= 31:
                    day_template["NumberIndex"].append(i)
                    day_template['Option'] += 1

            result_date = [None, None, None]
            all_numbers_c = deepcopy(all_numbers)

            for i in range(3):
                t_name = next_template(year_template, month_template, day_template)
                if t_name is None:
                    break
                targetIndex = {'year': 0, 'month': 1, 'day': 2}[t_name]

                template = templates[targetIndex]
                numberIndex = template["NumberIndex"][0]
                result_date[targetIndex] = all_numbers[numberIndex]
                all_numbers[numberIndex] = None

                for t in templates:
                    for j, n in enumerate(t["NumberIndex"]):
                        if n == numberIndex:
                            t["Option"] -= 1
                            t["NumberIndex"] = t["NumberIndex"][:j] + t["NumberIndex"][j+1:]
                            break

            while None in all_numbers:
                for i in range(len(all_numbers)):
                    if None is all_numbers[i]:
                        all_numbers = all_numbers[:i] + all_numbers[i+1:]
                        break

            if None in result_date and result_date[0] is not None:
                while None in result_date:
                    for i in [2, 1]:
                        if result_date[i] is None:
                            result_date[i] = all_numbers[0]
                            all_numbers = all_numbers[1:]
                            break
            elif result_date[0] is None:
                result_date = [all_numbers_c[2], all_numbers_c[0], all_numbers_c[1]]

            all_numbers = format_date(result_date)
            return numbers2str(all_numbers)

        def strip_data(data):
            '''
            去掉左右的干扰项
            '''
            data['company'] = data['company'].strip('a')# BAD ocr make some ' a' in the end of company
            pattern_1 = re.compile(r'^\d{4,}')# 去掉一些条码编号
            pattern_2 = re.compile(r'^\d\.')# 去掉一些1.XXXX,2.XXXX的数字编号
            for key,value in data.items():
                if key != 'items':
                    value = value.strip(' ')
                    value = value.strip('*')
                    data[key] = value.strip('.')
                else:
                    for i,item in enumerate(data['items']):
                        for k,v in item.items():
                            if k == 'name':
                                v = v.strip(' ')
                                start_num = pattern_1.findall(v)
                                start_id = pattern_2.findall(v)
                                for num in start_num:
                                    v = v.replace(num,'')
                                for id in start_id:
                                    v = v.replace(id,'')
                            v = v.strip('x') # BAD ocr
                            v = v.strip('*')
                            data['items'][i][k] = v.strip(' ')
            return data

        # 去掉左右干扰项
        data = strip_data(data)

        # 处理date
        # print(data['date'])
        if data['date'] == '':
            guess_date = guess_date_from_script(scripts)
            if guess_date != None:
                data['date'] = guess_date
            else:
                if ChineseData:
                    data['date'] = '2022-03-13' # 统计出来的最优值，所有数据中 2022-03-13 出现了 235 次
                else:
                    data['date'] = '2022-03-29' # 统计出来的最优值，所有数据中 2022-03-13 出现了 49 次
        else:
            data['date'] = parse_date(ChineseData,data['date'])

        # 处理total
        data['total'] = process_money(stringQ2B(data['total']))

        # 处理tax
        # 如果total字段有货币符号，则tax也有
        tax = stringQ2B(data['tax'])
        # print(tax)
        # if '$' in data['total']:
        #     tax+='$'
        #     print(tax)
        # elif '￥' in data['total']:
        #     tax+='￥'
        # 处理money类数字
        tax = tax.replace('O','.')
        tax = tax.replace('..','.')
        tax = tax.replace('%','.')
        data['tax'] = process_money(tax)

        # 处理company字段
        if data['company'] == '':
            if ChineseData:
                data['company'] = '一团火连锁团结公园店' # 中文小票出现最多的公司名
            else:
                data['company'] = 'Jack\'s Food' # 英文小票出现最多的公司名
        if '欢迎光临' in data['company']:
            data['company'] = data['company'].replace('欢迎光临','')
        elif 'Welcome' in ['company']:
            data['company'] = data['company'].replace('welcome','')
        for item in data['items']:
            # 处理name
            try:
                name = stringQ2B(item['name'])

                # 中文票据不包含空格
                if is_Chinese(name):
                    name = name.replace(' ','')
                
                # 去除小括号及中间部分
                index_1 = bracket(name,'(',')')
                name = name[:index_1[0]]+name[index_1[1]+1:]

                # 去除中括号及中间部分
                index_2 = bracket(name,'[',']')
                name = name[:index_2[0]]+name[index_2[1]+1:]

                # 去除大括号及中间部分
                index_3 = bracket(name,'{','}')
                name = name[:index_3[0]]+name[index_3[1]+1:]

                item['name'] = name
                # TODO: 如果name含货币单位, 需要将货币单位置于数字之前
            except:
                item['name'] = ''
            
            # 处理cnt
            try:
                item['cnt'] = '%.2f' % float(item['cnt'])
            except:
                item['cnt'] = '1.00'
            
            # 处理price
            try:
                item['price'] = process_money(item['price'])
            except:
                # 不存在单价字段，则按照total处理
                item['price'] = data['total']

        return data

if __name__=='__main__':
    ###############################################################################
    # 请根据实际路径修改下面两个变量
    model_path="/home/ma-user/work/model/fake_model.pth"
    data_path="./test_data.pkl"
    ###############################################################################

    ###############################################################################
    # 下方代码毋需改动
    infer_api=BaselineService("", model_path)

    file_content=open(data_path,"rb")
    request_data={
        "file1":{
            "test_data.pkl": file_content
        }
    }

    input_data=infer_api._preprocess(request_data)
    result=infer_api._inference(input_data)
    final_result=infer_api._postprocess(result)
    file_content.close()
    print(final_result)
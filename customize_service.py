# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
"""
import os
import torch
from copy import deepcopy
import re
from torchvision import transforms
from torchtext.data import Field, RawField
import pickle
import numpy as np

import cv2

from model.PICK.model import pick as pick_arch_model
from model.PICK.data_utils.documents import normalize_relation_features, sort_box_with_list
from model.PICK.utils.util import iob_index_to_str, text_index_to_str
from model.PICK.utils.class_utils import keys_vocab_cls


from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans
from model_service.pytorch_model_service import PTServingBaseService


class BaselineService(PTServingBaseService):

    # def __init__(self, model_name, model_path):
#         # 调用父类构造方法
#         super(BaselineService, self).__init__(model_name, model_path)
#         # 调用自定义函数加载模型

# class BaselineService:

    def __init__(self, model_name, model_path):
        # 调用父类构造方法
        # 调用自定义函数加载模型

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        
        dir_path = os.path.dirname(os.path.realpath(model_path))
        ckpt_path = os.path.join(dir_path, 'model_best.pth')
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        config = checkpoint['config']
        state_dict = checkpoint['state_dict']

        self.max_box_num = 300
        self.max_transcript_len = 50
        self.image_resize = (480, 960)
        self.entities = ["company", "date", "total", "tax", "name", "cnt", "price"]

        self.textSegmentsField = Field(sequential=True, use_vocab=True, include_lengths=True, batch_first=True)
        self.textSegmentsField.vocab = keys_vocab_cls

        self.trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.pick_model = config.init_obj('model_arch', pick_arch_model)
        self.pick_model = self.pick_model.to(self.device)
        self.pick_model.load_state_dict(state_dict)
        self.pick_model.eval()

    def _preprocess(self, request_data):
        data = None
        for k, v in request_data.items():
            for file_name, file_content in v.items():
                data = pickle.load(file_content) # use this when push code to huawei
                # data = file_content #use this in my own computer

        image = data['image']
        ocr = data['ocr']
        id = data['id']
        boxes_and_transcripts_data = [(i, box['points'], box['label']) for i, box in enumerate(ocr)]
        boxes_and_transcripts_data = sort_box_with_list(boxes_and_transcripts_data)
        boxes = []
        transcripts = []

        for _, points, transcript in boxes_and_transcripts_data:
            if len(transcript) == 0 :
                transcript = ' '
            mp = []
            for p in points:
                mp += p
            boxes.append(mp)
            transcripts.append(transcript)

        boxes_num = min(len(boxes), self.max_box_num)
        transcripts_len = min(max([len(t) for t in transcripts[:boxes_num]]), self.max_transcript_len)
        mask = np.zeros((boxes_num, transcripts_len), dtype=int)

        relation_features = np.zeros((boxes_num, boxes_num, 6))

        height, width, _ = image.shape

        image = cv2.resize(image, self.image_resize, interpolation=cv2.INTER_LINEAR)
        x_scale = self.image_resize[0] / width
        y_scale = self.image_resize[1] / height

        min_area_boxes = [cv2.minAreaRect(np.array(box, dtype=np.float32).reshape(4, 2)) for box in boxes[:boxes_num]]

        resized_boxes = []
        for i in range(boxes_num):
            box_i = boxes[i]
            transcript_i = transcripts[i]
            resized_box_i = [int(np.round(pos * x_scale)) if i%2 == 0 else int(np.round(pos * y_scale)) for i, pos in enumerate(box_i)]
            resized_rect_output_i = cv2.minAreaRect(np.array(resized_box_i, dtype=np.float32).reshape(4, 2))
            resized_box_i = cv2.boxPoints(resized_rect_output_i)
            resized_box_i = resized_box_i.reshape((8,))
            resized_boxes.append(resized_box_i)

            for j in range(boxes_num):
                transcript_j = transcripts[j]

                rect_output_i = min_area_boxes[i]
                rect_output_j = min_area_boxes[j]

                # Centers of rect_of_box_i and rect_of_box_j.
                center_i = rect_output_i[0]
                center_j = rect_output_j[0]

                width_i, height_i = rect_output_i[1]
                width_j, height_j = rect_output_j[1]

                # Center distances of boxes on x-axis.
                relation_features[i, j, 0] = np.abs(center_i[0] - center_j[0]) \
                    if np.abs(center_i[0] - center_j[0]) is not None else -1  # x_ij

                # Center distances of boxes on y-axis.
                relation_features[i, j, 1] = np.abs(center_i[1] - center_j[1]) \
                    if np.abs(center_i[1] - center_j[1]) is not None else -1  # y_ij

                relation_features[i, j, 2] = width_i / (height_i) \
                    if height_i != 0 and width_i / (height_i) is not None else -1  # w_i/h_i

                relation_features[i, j, 3] = height_j / (height_i) \
                    if height_i != 0 and height_j / (height_i) is not None else -1  # h_j/h_i

                relation_features[i, j, 4] = width_j / (height_i) \
                    if height_i != 0 and width_j / (height_i) is not None else -1  # w_j/h_i

                relation_features[i, j, 5] = len(transcript_j) / (len(transcript_i)) \
                    if len(transcript_j) / (len(transcript_i)) is not None else -1  # T_j/T_i

        relation_features = normalize_relation_features(relation_features, width=width, height=height)
        text_segments = [list(trans) for trans in transcripts[:boxes_num]]
        texts, texts_len = self.textSegmentsField.process(text_segments)
        texts = texts[:, :transcripts_len].cpu().numpy()
        texts_len = np.clip(texts_len.cpu().numpy(), 0, transcripts_len)
        text_segments = ( texts, texts_len)

        for i in range(boxes_num):
            mask[i, :texts_len[i]] = 1

        whole_image = RawField().preprocess(image)
        boxes_coordinate = RawField().preprocess(resized_boxes)
        relation_features = RawField().preprocess(relation_features)
        mask = RawField().preprocess(mask)
        boxes_num = RawField().preprocess(boxes_num)
        transcripts_len = RawField().preprocess(transcripts_len)

        return dict(
            whole_image=torch.stack([self.trsfm(whole_image)], dim=0).float().to(self.device),
            relation_features=torch.stack([torch.FloatTensor(relation_features)], dim=0).to(self.device),
            boxes_coordinate=torch.stack([torch.FloatTensor(boxes_coordinate)], dim=0).to(self.device),
            text_segments=torch.stack([torch.LongTensor(text_segments[0])], dim=0).to(self.device),
            text_length=torch.stack([torch.LongTensor(text_segments[1])], dim=0).to(self.device),
            mask=torch.stack([torch.ByteTensor(mask)], dim=0).to(self.device),
            file_id=[id],
            transcripts=transcripts
        )

    def _postprocess(self, data):
        data, original_input = data
        scripts = original_input['transcripts']
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
                all_numbers = pattern.findall(s)

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
                money = '%.2f' % float(money)
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
            all_numbers = pattern.findall(date)
            if len(all_numbers) < 3:
                return '2022-03-13' if is_chinese else '2022-03-29'

            if is_chinese or '-' in date:
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

            for i in range(3):
                t_name = next_template(year_template, month_template, day_template)
                if t_name is None:
                    break
                targetIndex = {'year': 0, 'month': 1, 'day': 2}[t_name]

                template = templates[targetIndex]
                numberIndex = template["NumberIndex"][0]
                result_date[targetIndex] = all_numbers[numberIndex]
                all_numbers = all_numbers[:numberIndex] + all_numbers[numberIndex+1:]

                for t in templates:
                    for j, n in enumerate(t["NumberIndex"]):
                        if n == numberIndex:
                            t["Option"] -= 1
                            t["NumberIndex"] = t["NumberIndex"][:j] + t["NumberIndex"][j+1:]
                            break

            if None in result_date:
                while None in result_date:
                    for i in [2, 1, 0]:
                        if result_date[i] is None:
                            result_date[i] = all_numbers[0]
                            all_numbers = all_numbers[1:]
                            break
            all_numbers = format_date(result_date)
            return numbers2str(all_numbers)



        # 处理date
        print(data['date'])
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
        # 处理money类数字

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

        if data['company'] == '':
            if ChineseData:
                data['company'] = '一团火连锁团结公园店' # 中文小票出现最多的公司名
            else:
                data['company'] = 'Jack\'s Food' # 英文小票出现最多的公司名
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

    def _inference(self, input_data):
        output = self.pick_model(**input_data)
        logits = output['logits']  # (B, N*T, out_dim)
        new_mask = output['new_mask']
        image_indexs = input_data['file_id']  # (B,)
        text_segments = input_data['text_segments']  # (B, num_boxes, T)
        mask = input_data['mask']
        # List[(List[int], torch.Tensor)]
        best_paths = self.pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
        predicted_tags = []
        for path, score in best_paths:
            predicted_tags.append(path)

        # convert iob index to iob string
        decoded_tags_list = iob_index_to_str(predicted_tags)
        # union text as a sequence and convert index to string
        decoded_texts_list = text_index_to_str(text_segments, mask)

        for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
            # List[ Tuple[str, Tuple[int, int]] ]
            spans = bio_tags_to_spans(decoded_tags, [])
            spans = sorted(spans, key=lambda x: x[1][0])

            entities = []  # exists one to many case
            for entity_name, range_tuple in spans:
                entity = dict(entity_name=entity_name,
                              text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                entities.append(entity)

            result = dict(
                company="",
                date="",
                total="",
                tax="",
                items=[]
            )
            new_item = dict()
            for e in entities:
                entity_name = e['entity_name']
                text = e['text']
                if entity_name in result:
                    result[entity_name] = text
                elif entity_name in ["price", "cnt", "name"]:
                    if entity_name in new_item:
                        if "cnt" not in new_item:
                            new_item['cnt'] = "1"
                        result['items'].append(new_item)
                        new_item = dict()
                    new_item[entity_name] = text

            return result, input_data
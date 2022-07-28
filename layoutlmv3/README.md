# LayoutLMv3 fine-tune
> 此为[ocr小票理解竞赛](https://competition.huaweicloud.com/information/1000041696/introduction)提交的根据LayoutLMv3模型进行fine-tune的相关代码

文档结构
```
.
├── README.md
├── data
│   └── train
├── dependency.py #用于生成提交文件中的环境依赖项
├── infer.py # 用于根据数据集进行一遍推理，得到推理结果（不可进行判分）
├── model
│   ├── config.json # 提交模型时需要
│   ├── customize_service.py # 提交模型时需要
│   ├── fake_model.pth # 提交模型时使用（帮助寻找到实际的模型路径）
│   ├── fine_tune.py # fine-tune阶段使用
│   ├── requirements.txt # 推理阶段时模型依赖环境
│   ├── saved # 模型权重
│   └── utils 
├── move.ipynb # 华为modelarts平台上和obs桶间转移
├── res_v3.json # 后处理后结果（一次debug的结果）
├── res_v3_wo_post.json # 后处理前结果（一次debug的结果）
└── test_data.pkl # 测试样本（来自竞赛指南）
```

其中模型权重和数据data由于过大未进行上传

fine-tune过程主要参考 https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3 的代码进行编写（打标过程为自己实现的）

但是此次fine-tune在B榜上效果欠佳，得分最终仅有35.6591

## fine-tune阶段的预处理过程如下：
> 思路：将KIE任务变为经过序列标注的分类任务

1. 读取数据集中的图片，ocr得到的text及对应的bbox的位置和该小票的ie\_label
2. 将ie\_label中的item项展平，得到一个仅有一个维度的字典gt
3. 根据gt为ocr结果中的每一个word打上标签
   1. 使用gt的数字与ocr得到的文本进行正则匹配，得到日期对应的位置
   2. company字段和name字段如果存在ocr的text和gt存在交集的字段，则视为对应的需要进行打标的位置
   3. 先去除掉一些ocr的text中的杂乱文本，利用正则匹配找到所有可能的位置，然后对可能位置处的字符按照和后处理一样的方法进行处理，如果处理后和gt完全匹配，则给对应位置打上money标签
4. 对图像大小进行归一化
5. 根据归一化后的图像的大小对bbox的坐标进行归一化，变为相对位置坐标
6. 使用huggingface中的预训练模型对应的processor对数据进行处理，得到模型的输入

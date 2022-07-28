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

fine-tune过程主要参考https://github.com/NielsRogge/Transformers-Tutorials/tree/master/LayoutLMv3的代码进行编写（打标过程为自己实现的）

但是此次fine-tune在B榜上效果欠佳，得分最终仅有35.6591
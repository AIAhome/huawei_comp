import datasets
from datasets import load_dataset,Features, Sequence, ClassLabel, Value, Array2D, Array3D
from transformers import LayoutLMv3ForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import wandb

# self implement
from model.utils.dataProcessor import MyDataProcessor,Entities_list,id2label,label2id
from model.utils.metric import compute_metrics



#-------
# Macro args
#-------
BATCH_SIZE = 50
TRAIN_TEST_SPLIT_RATIO = 0.8
EPOCHS = 5
TRAIN_EN = False
EVAL_STEPS = 300

# if TRAIN_EN:
#     wandb.init(project="LayoutLMv3 fine-tune-en")
# else:
#     wandb.init(project="LayoutLMv3 fine-tune-zh")

#-----------------
# prepare the dataset
#-----------------
if TRAIN_EN:
    dataset = load_dataset("model/utils/my_dataset.py",'en',data_dir='data')
else:
    dataset = load_dataset('model/utils/my_dataset.py','zh',data_dir='data')

logger = datasets.logging.get_logger(__name__)
logger.info(dataset)

processor = MyDataProcessor(train_en=True)# TODO: maybe a bug in huggingface code??? when i specify the language, there will be a bug

# we need to define custom features for `set_format` (used later on) to work properly
features = Features({
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),# the LayoutLMv3 processor output pixel_values to be (3,224,224) instead of (3,resized_size[0],resized_size[1])
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),# [168, 103, 550, 136]
    'labels': Sequence(feature=Value(dtype='int64')),# [-100, 1, -100, 2, 2, 0, 0, 0, -100, -100, -100, 0, 0, -100, ...]
})

dataset = dataset["train"].map(
    processor.process_example,
    batched=True,
    remove_columns=dataset["train"].column_names,
    features=features,
    batch_size=BATCH_SIZE
)

dataset.set_format("torch")

dataset = dataset.train_test_split(train_size=TRAIN_TEST_SPLIT_RATIO)
train_dataset = dataset['train']
test_dataset = dataset['test']

#-----------
# define the model
#-----------
if TRAIN_EN:
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",id2label=id2label,label2id=label2id)
    save_dir = 'model/saved/en'
else:
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base-chinese",id2label=id2label,label2id=label2id)
    save_dir = 'model/saved/zh'


training_args = TrainingArguments(output_dir=save_dir,
                                  num_train_epochs=EPOCHS,
                                  logging_steps=EVAL_STEPS,
                                  per_device_train_batch_size=2,
                                  per_device_eval_batch_size=2,
                                  evaluation_strategy="steps",
                                  eval_steps=EVAL_STEPS,
                                  save_strategy='epoch',
                                  save_total_limit=5,
                                #   load_best_model_at_end=True,metric_for_best_model="f1",
                                  report_to="wandb")

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=processor.processor,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
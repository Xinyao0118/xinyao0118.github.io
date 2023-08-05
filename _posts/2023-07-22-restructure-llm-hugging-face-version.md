---
layout:     post
title:      LLM Science Exam
subtitle:   A ML pipeline for building a Large Language Model
date:       2023-07-22
author:     Xinyao Wu
header-img: img/home-bg.jpg
catalog: true
tags:
    - Machine Learning
    - LLM
    - NLP
    - AI
---
<img src="/img/LLM-Kaggle.png"/>

```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForMultipleChoice,TrainingArguments, Trainer
```

    /opt/conda/lib/python3.10/site-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.23.5
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    /opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:98: UserWarning: unable to load libtensorflow_io_plugins.so: unable to open file: libtensorflow_io_plugins.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so']
    caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io_plugins.so: undefined symbol: _ZN3tsl6StatusC1EN10tensorflow5error4CodeESt17basic_string_viewIcSt11char_traitsIcEENS_14SourceLocationE']
      warnings.warn(f"unable to load libtensorflow_io_plugins.so: {e}")
    /opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/__init__.py:104: UserWarning: file system plugins are not loaded: unable to open file: libtensorflow_io.so, from paths: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/opt/conda/lib/python3.10/site-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZTVN10tensorflow13GcsFileSystemE']
      warnings.warn(f"file system plugins are not loaded: {e}")



```python
# Following datacollator (adapted from https://huggingface.co/docs/transformers/tasks/multiple_choice)
# will dynamically pad our questions at batch-time so we don't have to make every question the length
# of our longest question.

from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
```


```python
# --- skeleton Requirement: can generate different predictions for different models in Hugging Face.

class LLM_prediction:

    def __init__(self,model_path,options = 'ABCDE'):
        self.model_path = model_path
        self.options = options
        self.indices = list(range(len(options)))
        self.option_to_index = {option: index for option, index in zip(self.options, self.indices)}
        self.index_to_option = {index: option for option, index in zip(self.options, self.indices)}
        return

    def read_data(self,data_folder = None):
        #training
        train_df = pd.read_csv(f"{data_folder}/train.csv")
        self.train_ds = Dataset.from_pandas(train_df)
        #testing
        self.test_df = pd.read_csv(f"{data_folder}/test.csv")
        return self.train_ds,self.test_df

    def pre_process_data(self,row):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        question = [row['prompt']]*5
        answers = []
        for option in self.options:
            answers.append(row[option])
        tokenized_row = self.tokenizer(question,answers,truncation = True)
        tokenized_row['label'] = self.option_to_index[row['answer']]
        return tokenized_row

    def nlp(self,output_model_dir = 'finetuned_bert'):
        #return trainer
        model = AutoModelForMultipleChoice.from_pretrained(self.model_path)
        tokenized_train_ds = self.train_ds.map(self.pre_process_data, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
        training_args = TrainingArguments(
            output_dir=output_model_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to='none'
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_train_ds,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForMultipleChoice(tokenizer=self.tokenizer),
        )
        self.trainer.train()
        return self.trainer

    def predictions_to_map_output(self,predictions):
        sorted_answer_indices = np.argsort(-predictions)
        top_answer_indices = sorted_answer_indices[:,:3] # Get the first three answers in each row
        top_answers = np.vectorize(self.index_to_option.get)(top_answer_indices)
        return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)

    def inference(self,assign_random_answer = True):
        if not assign_random_answer:
            raise ValueError('Another inference way has not been be developed.')
        self.test_df['answer']='A'
        self.test_ds = Dataset.from_pandas(self.test_df)
        tokenized_test_ds = self.test_ds.map(self.pre_process_data, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
        predictions=self.trainer.predict(tokenized_test_ds)
        return self.predictions_to_map_output(predictions.predictions)



```


```python
llm_test = LLM_prediction(model_path = '/kaggle/input/huggingface-bert/bert-base-cased')
```


```python
train_ds,test_ds = llm_test.read_data(data_folder ='/kaggle/input/kaggle-llm-science-exam')
```


```python
llm_test.nlp()
```

    Some weights of the model checkpoint at /kaggle/input/huggingface-bert/bert-base-cased were not used when initializing BertForMultipleChoice: ['cls.seq_relationship.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.seq_relationship.bias', 'cls.predictions.bias']
    - This IS expected if you are initializing BertForMultipleChoice from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMultipleChoice from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForMultipleChoice were not initialized from the model checkpoint at /kaggle/input/huggingface-bert/bert-base-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

    /opt/conda/lib/python3.10/site-packages/transformers/optimization.py:411: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(
    You're using a BertTokenizerFast tokenizer. Please note that with a fast tokenizer, using the `__call__` method is faster than using a method to encode the text followed by a call to the `pad` method to get a padded encoding.
    /opt/conda/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
      warnings.warn('Was asked to gather along dimension 0, but all '




    <div>

      <progress value='75' max='75' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [75/75 00:54, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>No log</td>
      <td>1.480417</td>
    </tr>
    <tr>
      <td>2</td>
      <td>No log</td>
      <td>1.163188</td>
    </tr>
    <tr>
      <td>3</td>
      <td>No log</td>
      <td>1.066047</td>
    </tr>
  </tbody>
</table><p>


    /opt/conda/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
      warnings.warn('Was asked to gather along dimension 0, but all '
    /opt/conda/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
      warnings.warn('Was asked to gather along dimension 0, but all '





    <transformers.trainer.Trainer at 0x7ebc65122440>




```python
res = llm_test.inference()
```

    Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
    /opt/conda/lib/python3.10/site-packages/torch/nn/parallel/_functions.py:68: UserWarning: Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.
      warnings.warn('Was asked to gather along dimension 0, but all '







```python
submission_df = test_ds[['id']]
submission_df['prediction'] = res
```

    /tmp/ipykernel_28/3749572995.py:2: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      submission_df['prediction'] = res



```python
submission_df.to_csv('submission.csv', index=False)
```

# [Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs](https://arxiv.org/abs/2503.08551)

In this repository, we present the code to our paper "Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs" by Wanyong Feng, Peter Tran, Stephen Sireci, and Andrew Lan. In this work, we propose a novel method to predict the difficulty value of math MCQs. Our method first augments the reasoning steps and feedback messages for each option, then samples student knowledge profiles from a distribution, and finally predicts the likelihood of each option being selected and use it to predict the MCQ’s difficulty.

For any questions please [email](mailto:wanyongfeng@umass.edu) or raise an issue.

## Dataset
For MAPT dataset, we cannot share the data since it is a private dataset
For EEDI dataset, we manually parse the question text and options from the images and filter out questions that need images to answer the question. This is also a private dataset. You can [email](mailto:wmcnichols@umass.edu) for access

## Running

### Setup
```
python -m venv mcq_diff_prediction_env
source mcq_diff_prediction_env/bin/activate
python -m pip install -r requirements.txt
```

### Generate reasonings for each option
```
python reason_prompt_gen.py
python reasoning_prompting.py
python reasoning_post_process.py
```

### Linear regression baseline

#### Extract features
```
python feature_extract.py
```

#### Linear regression close-form solution
```
python regression_close_form.py
```

### Finetune with/without reasoning
```
python LLM_pred_difficulty.py
```

### Our method
```
python our_method.py
```


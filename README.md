# CGMIA

# Keep Evaluation Fair: Detecting Data Leakage in Code Generation Benchmarks via Membership Inference Attacks


This repository contains the implementation code for the paper "Keep Evaluation Fair: Detecting Data Leakage in Code Generation Benchmarks via Membership Inference Attacks". The code focuses on membership inference attacks to identify data leakage in code generation benchmarks, leveraging multi-modal features (code semantics, expert metrics) and deep learning models for classification.



## Environment Setup

```
pip install -r requirement.txt
```

## Dataset

The datasets we used are stored in the "`data`" file. The datasets in Java language and Python language are respectively saved in two separate files.

## Finetune

You can use the `finetune/finetune.py` file to fine-tune the target model. And modify the `model_name` in `finetune.py`.

```python
model_name = "deepseek-coder-1.3b-instruct"
model_path = f"./models/{model_name}"
tokenizer_path = f"./models/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,device_map="auto")
```

## Generate

You can use the `finetune/generate/generate_all.py` file to perform inference on the fine-tuned model and generate code. Please note that you need to modify the `model_name` in the code to be consistent with the fine-tuned model.

## MIA

In the `MIA` folder, there are all the methods mentioned in our paper regarding MI. The methods we proposed are in the `MIA/cgmia.py` file. 
Before running the MIA code file, the generated code needs to be preprocessed.

Place {model_name}_data.json in the parent directory of the code script (or modify the file path in with open("../{model_name}_data.json","r",encoding="utf-8") as f: to match your data location).

The code is organized into 5 core modules. Below is a breakdown of each module's purpose and key parameters.


The input file {model_name}_data.json (stored in the parent directory of the code script) must follow this schema for each entry:

```python
dict_keys(['answer', 'benchmark', 'class', 'code', 'code_vec', 'index', 'lev', 'model', 'perplexity', 'task_id', 'vec_dis', 'ranks'])
```

| Key          | Type       | Description                                                                 |
|--------------|------------|-----------------------------------------------------------------------------|
| `model`      | String     | Name of the code generation model (e.g., "phi-2").                          |
| `task_id`    | String/Int | Unique ID of the code generation task.                                      |
| `bench`      | String     | Name of the benchmark (e.g., "HumanEval", "MBPP").                          |
| `index`      | Int        | Index of the entry in the original raw data.                                |
| `perplexity` | Float      | Perplexity of the model’s output (measure of prediction uncertainty).       |
| `lev`        | Float      | Levenshtein distance between the model’s output and the reference solution. |
| `code_vec`   | List[Float]| Vector embedding of the model’s generated code (e.g., from CodeBERT).       |
| `pass`       | Int        | Binary indicator (0/1) for whether the generated code passes unit tests.    |
| `codebleu`   | Float      | CodeBLEU score (metric for code generation quality).                        |
| `solution_vec`| List[Float]| Vector embedding of the reference solution code.                            |
| `class`      | Int        | Binary label for membership (1 = "member" of the training set; 0 = "non-member"). |


## Ensemble Learning
The core goal of this code is to improve the robustness of membership inference attacks by leveraging ensemble voting—combining predictions from multiple models to reduce bias and improve generalization. The workflow includes:
1.Loading and preprocessing code generation benchmark data (model outputs, metrics, and embeddings).
2.Training three independent models (each on distinct datasets) to classify "member" vs. "non-member" samples.
3.Generating predictions from each trained model on a shared test set.
4.Applying majority voting to aggregate predictions and calculating evaluation metrics.
5.Saving detailed results and metrics for analysis.
The Ensemble Learning method we proposed are in the `MIA/Ensemble_learning.py` file. 

## Hardware Requirements
GPU (Recommended): NVIDIA GPU with ≥8GB VRAM (for efficient training of convolutional and fully connected layers).
CPU (Fallback): Supported, but training will be significantly slower (expect longer runtime for large datasets).


## APPS data
You can obtain the APPS dataset from the link below.
https://github.com/Xin-Zhou-smu/LessLeak-bench
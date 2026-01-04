import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import os
import gc
import time
from datetime import datetime

benchmark_1 = "../../data/java/humaneval_java.jsonl"
benchmark_2 = "../../data/java/mbjp_release_v1.2.jsonl"
benchmark_3 = "../../data/java/ncb_java_en.jsonl"
benchmark_4 = "../../data/python/HumanEval.jsonl"
benchmark_5 = "../../data/python/mbpp_release_v1.jsonl"
benchmark_6 = "../../data/python/ncb_python_en.jsonl"
benchmark_7 = "../../data/python/ClassEval_data.json"
benchmark_8 = "../../data/python/evo_data.json"

all_benchmark = [benchmark_1,benchmark_2,benchmark_3,benchmark_4,benchmark_5,benchmark_6,benchmark_7,benchmark_8]

# Create directory for runtime logs
os.makedirs("../logs/runtime_logs", exist_ok=True)


def generate_code(prompt, max_length,tokenizer, p_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # Reduce memory usage
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = p_model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.5,
            do_sample=True,
        )
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Explicitly release intermediate tensors
        del inputs
        del outputs
        torch.cuda.empty_cache()

        return generated_code


def generate_1(benchmark,model_name,index,bench_name):
    start_time = time.time()  # Record start time
    test_data = []
    if bench_name=="class_eval" or bench_name=="evo_eval":
        with open(benchmark, "r", encoding="utf-8") as file:
            test_data = json.load(file)
    else:
        with open(benchmark, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line.strip())
                test_data.append(data)

    tokenizer_path = f"../models/{model_name}"
    model_path = f"../models/{model_name}"
    directory = f"../code/{model_name}/{bench_name}/{index}/"

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    try:
        for i in tqdm(range(len(test_data))):
            if bench_name == "class_eval":
                prompt = "Please complete the "+ test_data[i]["class_name"] +\
                         " function in the given Python code.\n\nInput Code:\n```Python\n"\
                         +test_data[i]["skeleton"]+"\n```\nCompleted Code:"
            else:
                prompt = test_data[i]["prompt"]
            # Generate code
            code = generate_code(
                prompt=prompt,
                max_length=len(prompt),
                p_model=model,
                tokenizer=tokenizer
            )

            # Save results
            if bench_name == "ncb_java" or bench_name == "ncb_python":
                file_name = str(test_data[i]["_id"]).replace("/", "")
            elif bench_name == "evo_eval":
                file_name = str(test_data[i]["namespace"]).replace("/", "")
            else:
                file_name = str(test_data[i]["task_id"]).replace("/", "")
            save_path = os.path.join(directory, f"{file_name}.txt")
            with open(save_path, 'w') as f:
                f.write(code)

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        # Release resources
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate time taken
     # Log runtime to file
    log_entry = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Benchmark: {bench_name}, Model: {model_name}, Iteration: {index}, Time: {elapsed_time:.2f} seconds\n"

    log_file = f"../logs/runtime_logs/runtime_{bench_name}_{model_name}.txt"
    with open(log_file, 'a') as f:
        f.write(log_entry)

    return elapsed_time

if __name__ == '__main__':
    model_names = ["codegemma-2b","deepseek-coder-1.3b-instruct","Qwen2.5-Coder-3B-Instruct"]

    max_generate_nums = 1
    benchmark_1 = "../../data/java/humaneval_java.jsonl"
    benchmark_2 = "../../data/java/mbjp_release_v1.2.jsonl"
    benchmark_3 = "../../data/java/ncb_java_en.jsonl"
    benchmark_4 = "../../data/python/HumanEval.jsonl"
    benchmark_5 = "../../data/python/mbpp_release_v1.jsonl"
    benchmark_6 = "../../data/python/ncb_python_en.jsonl"
    benchmark_7 = "../../data/python/ClassEval_data.json"
    benchmark_8 = "../../data/python/evo_data.json"

    bench_names = {"../../data/java/humaneval_java.jsonl": "humaneval_java",
                   "../../data/java/mbjp_release_v1.2.jsonl": "mbxp_java",
                   "../../data/java/ncb_java_en.jsonl": "ncb_java",
                   "../../data/python/HumanEval.jsonl": "humaneval_python",
                   "../../data/python/mbpp_release_v1.jsonl": "mbxp_python",
                   "../../data/python/ncb_python_en.jsonl": "ncb_python",
                   "../../data/python/ClassEval_data.json": "class_eval",
                   "../../data/python/evo_data.json": "evo_eval"}

    total_start = time.time()
    for bench in all_benchmark:
        for llm_model in model_names:
            for num in range(max_generate_nums):
                runtime = generate_1(benchmark=bench,
                                     model_name=llm_model,
                                     index=1,
                                     bench_name=bench_names[bench])
                print(f"Completed: {bench_names[bench]} - {llm_model} - Iter {num + 1} in {runtime:.2f}s")
    total_end = time.time()
    total_time = total_end - total_start
    # Log total runtime
    with open("../logs/runtime_logs/total_runtime.txt", 'a') as f:
        f.write(f"\nTotal execution time: {total_time:.2f} seconds "
                f"({total_time / 3600:.2f} hours)\n")
        f.write(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
# Poetic LLM Training Process

## Catalog
1. [Data Preparation](#data-preparation)
2. [Model Selection](#model-selection)
3. [Model Training](#model-training)
4. [Model quantization](#model-quantization)
5. [Model Evaluation](#model-evaluation)
6. [Model deployment](#model-deployment)
7. [APPENDIX1: Qwen1.5-7B-Chat series Inference performance](#appendix1-qwen15-7b-chat-series-inference-performance)
8. [APPENDIX2: Qwen1.5-7B-Chat Fine-tuning](#appendix2-qwen15-7b-chat-fine-tuning)
9. [APPENDIX3: Training commands with best performance](#appendix3-training-commands-with-best-performance)
10. [APPENDIX4: Dataset Processing](#appendix4-dataset-processing)


### 1. Data Preparation 

   - [Chinese poetry](https://github.com/chinese-poetry/chinese-poetry): Traditional Chinese (need to convert traditional Chinese into simplified Chinese)

   - [Poetry](https://github.com/Werneror/Poetry): Simplified Chinese

   - Dataset format: We choose poems with a title of 2-6 hanzis to construct the dataset, json and jsonl are permitted, then trimmed it into following format about 20000 poems: {"query": "11111", "response": "22222"}

     For dataset processing details see ***Appendix 4***.

### 2. Model Selection: [Qwen-1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat)

   - Powerful basic LLM capability outperforms other open-source LLMs in the [SurperCLUE](https://www.superclueai.com/) and [lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) leaderboard.

   - Chinese corpus optimization and extension of word list for Chinese poems.

   - Available gptq,awq method Int4 quantization, after quantization the GPU memory is about 8.2GB for inference.

     For model performance see ***Appendix 1***.

### 3. Model Training

Fine-tuning methods with [swift](https://github.com/modelscope/swift) infrastructure: LoRA, QLoRA and GaLore.

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA:Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2403.03507)

We use these fine-tuning methods, changing hyperparameters to get the best results. Here we find the loss has converged during training Qwen1.5-7B-Chat. After test, we found no evidence of overfitting.

![train_loss](D:\github\swift\output\qwen1half-7b-chat\v11-20240416-113331\images\train_loss.png)

For training details and best supervised fine-tuning parameters for different models, see ***Appendix 2*** and ***Appendix 3***

### 4. Model  quantization: 

- Step one: merge lora with `optimizer.pt`
- Step two: quantization 4 with AutoGPTQ and AWQ-Kernel

```
# merge lora only
$env:CUDA_VISIBLE_DEVICES="0"
swift export `
    --ckpt_dir "D:\github\swift\output\qwen1half-7b-chat-awq\v6-20240419-173504\checkpoint-984" `
    --merge_lora $true
```

```
# AWQ-INT4 quantization (takes about 24 minutes using A5000, memory usage: 23GB)
$env:CUDA_VISIBLE_DEVICES="0"
swift export `
    --ckpt_dir "D:\github\swift\output\qwen1half-7b-chat\v14-20240503-160444\checkpoint-1875" `
    --merge_lora $true `
    --quant_bits 4 `
    --load_dataset_config $true `
    --quant_method awq `
    --quant_n_samples 256 `
    --quant_seqlen 2048
```

```
# GPTQ-INT4 quantization
$env:CUDA_VISIBLE_DEVICES="0"
swift export `
    --ckpt_dir "D:\github\swift\output\qwen1half-7b-chat\v11-20240416-113331\checkpoint-1875" `
    --merge_lora $true `
    --quant_bits 4 `
    --load_dataset_config $true `
    --quant_method gptq `
    --quant_n_samples 256 `
    --quant_seqlen 1024
```

If OOM occurs during quantization, you can appropriately reduce `--quant_n_samples` (default 256) and `--quant_seqlen` (default 1024).

inference quantized model:

```
$env:CUDA_VISIBLE_DEVICES="0"
swift infer `
    --ckpt_dir "D:\github\swift\output\qwen1half-7b-chat\v11-20240416-113331\checkpoint-1875-merged-awq-int4"
```

```
# cmd in venv of qwen
D:\github\ENV\qwen\Scripts\python.exe d:\github\swift\swift\cli\infer.py --ckpt_dir D:\github\swift\output\qwen1half-moe-a2_7b-chat-int4\v1-20240426-164324\checkpoint-3939
```

GPU memory: about 8.5GB, so slow

deploy quantized llm(server)

```
$env:CUDA_VISIBLE_DEVICES="0"
swift deploy `
    --ckpt_dir "D:\github\swift\output\qwen1half-7b-chat\v3-20240329-172932\checkpoint-3936-merged-awq-int4"
```

test:

```
curl http://localhost:8000/v1/chat/completions `
    -H "Content-Type: application/json" `
    -d '{
    "model": "qwen1half-7b-chat",
    "messages": [{"role": "user", "content": "请根据这些关键词'炊烟 青砖 蝉鸣 苍松'，以'荒'为韵脚，生成一首以'石钟山'为主题的五言律诗。五言律诗格式为8小句，每小句字数为5个字，共40个字。"}],
    "max_tokens": 512,
    "temperature": 0
    }'
```

### 5. Model Evaluation

We chose the validation dataset in the same format compared to training dataset. However, we found that the content of the validation dataset was based on ancient poems and did not fully meet the requirements nowadays. Because in many of the company's exhibitions, poetry requires green and positive energy and is in line with the core socialist values. Based on this, we took the poetry of the Jiujiang project(九江乡村振兴科技馆-捧着课本游九江) as an example, including image, scenic spots, rhyme and poet style to create poetry. Under such a verification set about 1200 queries: 300 wujue, 300 wulv, 300 qijue, 300 qilv, we obtained the following result: 

- WuJue Accuracy Rate: 97.33%
- WuLv Accuracy Rate: 79.67%
- QiJue Accuracy Rate: 93.67%
- QiLv Accuracy Rate: 75.00%
- Total Accuracy Rate: 86.42%
- Time taken: 1252.1988081932068s

Compared with the model before quantization(bf16), the total accuracy error is less than 1%, the total accuracy rate of bf16 model with best performance is about 87.08%. There is no significant difference in terms of model generation quality before and after quantization.

In the inference stage after the end of int4 quantization, we found that the speed of the two quantization methods is much slower than the bf16 model inference. For speed problems, it may be that the inverse quantization process is slow, but it can not rule out inference engine optimization and hardware support. For awq quantification, we have found a way to install the awq acceleration core to increase the speed, but there is no way to increase the speed of gptq for the time being, which means that for 1200 poems generation, it takes about 15000s to obtain similar results.

### 6. Model deployment: vllm or Triton(linux only) inference acceleration

   vllm, tensorRT, triton, openvino



### APPENDIX1: Qwen1.5-7B-Chat series Inference performance:

Speed and GPU Memory Usage for generating 2048 tokens.(index not suitable for fine-tuned models)

| Quantization | Speed(Tokens/s) | GPU Memory Usage(GB) |
| ------------ | ----- | ---------------- |
| BF16 |  40.93|16.99|
|Int8|37.47|11.20|
|Int4|50.09|8.21|

Ps: quantization package: AutoGPTQ, AWQ

In conclusion, we choose Int4 model with about 8GB GPU memory usage.



### APPENDIX2: Qwen1.5-7B-Chat Fine-tuning

Common parameters settings: CUDA 11.8, Pytorch 2.0, flash attention 2, batch_size=1, gradient accumulation = 8.


| Method           | #Nodes | Sequence Length 256 | 512              | 1024             | 2048             | 4096              | 8192              |
| ---------------- | ------ | ------------------- | ---------------- | ---------------- | ---------------- | ----------------- | ----------------- |
| LoRA             | 1      | 20.1G / 1.2s/it     | 20.4G / 1.5s/it  | 21.5G / 2.8s/it  | 23.8G / 5.2s/it  | 29.7G / 10.1s/it  | 36.6G / 21.3s/it  |
| Q-LoRA           | 1      | 11.5G / 3.0s/it     | 11.5G / 3.0s/it  | 12.3G / 3.5s/it  | 13.9G / 7.0s/it  | 16.9G / 11.6s/it  | 23.5G / 22.3s/it  |
| Full-parameter   | 2      | 139.2G / 4.0s/it    | 148.0G / 4.0s/it | 162.0G / 4.5s/it | -                | -                 | -                 |

To sum up, LoRA with sequence_length==1024 and QLoRA methods are appropriate for our training.



### APPENDIX3: Training commands with best performance

1. Qwen1.5-7B-Chat sft

```
# lora: takes about 3.5h using A5000, memory usage: 23GB
$env:PYTHONPATH="C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
$env:CUDA_VISIBLE_DEVICES="0"
python llm_sft.py `
    --model_type qwen1half-7b-chat `
    --model_id_or_path "D:\poem_model\Qwen1.5-7B-Chat" `
    --sft_type lora `
    --tuner_backend peft `
    --dtype AUTO `
    --output_dir "D:\github\swift\output" `
    --train_dataset_sample 19000 `
    --max_length 1024 `
    --check_dataset_strategy warning `
    --lora_rank 8 `
    --lora_alpha 32 `
    --lora_dropout_p 0.05 `
    --lora_target_modules ALL `
    --gradient_checkpointing true `
    --batch_size 2 `
    --num_train_epochs 3 `
    --weight_decay 0.1 `
    --learning_rate 3e-5 `
    --gradient_accumulation_steps 16 `
    --max_grad_norm 0.5 `
    --warmup_ratio 0.05 `
    --eval_steps 50 `
    --save_steps 50 `
    --save_total_limit 3 `
    --logging_steps 10 `
    --use_flash_attn false `
    --self_cognition_sample 1000 `
    --model_name 风语诗人 'FengPoet' `
    --model_author 高钰博 'Geoffery' `
    --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
    --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
```

```
# sft galore
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type full \
    --use_galore true \
    --galore_update_proj_gap 400 \
    --train_dataset_sample -1 \
    --eval_steps 100 \
    --output_dir output \
    --num_train_epochs 3 \
    --max_length 1024 \
    --learning_rate 3e-5 \
    --use_flash_attn false \
    --save_only_model true \
    --preprocess_num_proc 4 \
    --model_name 风语诗人 'FengPoet' \
    --model_author 高钰博 'Geoffery' \
    --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" \
    --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl" \
```

2. Qwen1.5-7B-Chat-AWQ sft

   ```
   $env:PYTHONPATH="C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
   $env:CUDA_VISIBLE_DEVICES="0"
   python llm_sft.py `
       --model_type qwen1half-7b-chat-awq `
       --model_id_or_path "D:\poem_model\Qwen1.5-7B-Chat-AWQ" `
       --sft_type lora `
       --tuner_backend peft `
       --dtype AUTO `
       --output_dir "D:\github\swift\output" `
       --train_dataset_sample -1 `
       --num_train_epochs 3 `
       --max_length 1024 `
       --check_dataset_strategy warning `
       --use_loss_scale true `
       --lora_rank 8 `
       --lora_alpha 32 `
       --lora_dropout_p 0.05 `
       --lora_target_modules ALL `
       --gradient_checkpointing true `
       --batch_size 8 `
       --weight_decay 0.1 `
       --learning_rate 3e-5 `
       --gradient_accumulation_steps 8 `
       --max_grad_norm 1.0 `
       --warmup_ratio 0.05 `
       --eval_steps 50 `
       --save_steps 50 `
       --save_total_limit 3 `
       --logging_steps 10 `
       --use_flash_attn false `
       --self_cognition_sample 1000 `
       --model_name 风语诗人 'FengPoet' `
       --model_author 吴文俊 'Wendy' `
       --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
       --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
   ```

3.  Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4

   ```
   # 17GB GPU memory
   D:\github\ENV\qwen\Scripts\python.exe d:\github\swift\swift\cli\sft.py `
       --model_type qwen1half-moe-a2_7b-chat-int4 `
       --model_id_or_path "D:\models\Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4" `
       --sft_type lora `
       --dtype AUTO `
       --output_dir "D:\github\swift\output" `
       --train_dataset_sample -1 `
       --num_train_epochs 3 `
       --max_length 1024 `
       --check_dataset_strategy warning `
       --lora_rank 8 `
       --lora_alpha 32 `
       --lora_dropout_p 0.05 `
       --lora_target_modules ALL `
       --gradient_checkpointing true `
       --batch_size 16 `
       --weight_decay 0.1 `
       --learning_rate 2e-5 `
       --gradient_accumulation_steps 1 `
       --max_grad_norm 1.0 `
       --warmup_ratio 0.03 `
       --eval_steps 50 `
       --save_steps 50 `
       --save_total_limit 3 `
       --logging_steps 10 `
       --use_flash_attn false `
       --self_cognition_sample 1000 `
       --model_name 风语诗人 'FengPoet' `
       --model_author 高钰博 'Geoffery' `
       --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
       --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
   ```

4. Qwen1.5-MoE-A2.7B-Chat

   ```
   # 42GB GPU memory required
   $env:PYTHONPATH="C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
   $env:CUDA_VISIBLE_DEVICES="0"
   python llm_sft.py `
       --model_type qwen1half-moe-a2_7b-chat `
       --model_id_or_path "D:\models\Qwen1.5-MoE-A2.7B-Chat" `
       --sft_type lora `
       --tuner_backend peft `
       --dtype AUTO `
       --output_dir "D:\github\swift\output" `
       --train_dataset_sample -1 `
       --num_train_epochs 3 `
       --max_length 512 `
       --check_dataset_strategy warning `
       --use_loss_scale true `
       --lora_rank 8 `
       --lora_alpha 32 `
       --lora_dropout_p 0.05 `
       --lora_target_modules ALL `
       --gradient_checkpointing true `
       --batch_size 1 `
       --weight_decay 0.1 `
       --learning_rate 5e-5 `
       --gradient_accumulation_steps 4 `
       --max_grad_norm 1.0 `
       --warmup_ratio 0.03 `
       --eval_steps 50 `
       --save_steps 50 `
       --save_total_limit 3 `
       --logging_steps 10 `
       --use_flash_attn false `
       --self_cognition_sample 1000 `
       --model_name 风语诗人 'FengPoet' `
       --model_author 高钰博 'Geoffery' `
       --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
       --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
   ```

5. llama3-8B sft

   ```
   # GPU Mmemory: 23GB
   $env:CUDA_VISIBLE_DEVICES="0"
   swift sft `
       --model_type llama3-8b-instruct `
       --model_id_or_path 'D:\models\Meta-Llama-3-8B-Instruct' `
       --model_revision "master" `
       --sft_type "lora" `
       --tuner_backend "peft" `
       --template_type "llama3" `
       --dtype "AUTO" `
       --output_dir "D:\github\swift\output" `
       --train_dataset_sample -1 `
       --num_train_epochs 3 `
       --max_length 1024 `
       --check_dataset_strategy "warning" `
       --lora_rank 8 `
       --lora_alpha 32 `
       --lora_dropout_p 0.05 `
       --lora_target_modules "ALL" `
       --gradient_checkpointing $true `
       --batch_size 2 `
       --weight_decay 0.1 `
       --learning_rate 2e-5 `
       --gradient_accumulation_steps 16 `
       --max_grad_norm 0.5 `
       --warmup_ratio 0.05 `
       --eval_steps 50 `
       --save_steps 50 `
       --save_total_limit 3 `
       --logging_steps 10 `
       --model_name 风语诗人 'FengPoet' `
       --model_author 吴文俊 'Wendy' `
       --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
       --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
   ```

   ```
   # llama3-8b-instruct infer
   $env:CUDA_VISIBLE_DEVICES="0"; swift infer `
       --ckpt_dir "D:\github\swift\output\llama3-8b-instruct\v0-20240424-134014\checkpoint-1875" `
       --load_dataset_config true `
       --max_new_tokens 2048 `
       --temperature 0.1 `
       --top_p 0.7 `
       --repetition_penalty 1.0 `
       --do_sample true `
       --merge_lora $false
   ```

6. OpenBuddy/openbuddy-llama3-8b-v21.1-8k

   ```
   # sft GPU Memory: 23GB
   $env:CUDA_VISIBLE_DEVICES="0"
   swift sft `
       --model_type openbuddy-llama3-8b-chat `
       --model_id_or_path 'D:\models\Meta-Llama-3-8B-Instruct' `
       --model_revision "master" `
       --sft_type "lora" `
       --tuner_backend "peft" `
       --template_type "llama3" `
       --dtype "AUTO" `
       --output_dir "D:\github\swift\output" `
       --train_dataset_sample -1 `
       --num_train_epochs 3 `
       --max_length 1024 `
       --check_dataset_strategy "warning" `
       --lora_rank 8 `
       --lora_alpha 32 `
       --lora_dropout_p 0.05 `
       --lora_target_modules "ALL" `
       --gradient_checkpointing $true `
       --batch_size 2 `
       --weight_decay 0.1 `
       --learning_rate 2e-5 `
       --gradient_accumulation_steps 16 `
       --max_grad_norm 0.5 `
       --warmup_ratio 0.05 `
       --eval_steps 50 `
       --save_steps 50 `
       --save_total_limit 3 `
       --logging_steps 10 `
       --model_name 风语诗人 'FengPoet' `
       --model_author 吴文俊 'Wendy' `
       --custom_train_dataset_path "D:\dataset\poems_processed\rhyme\train_d1.jsonl" `
       --custom_val_dataset_path "D:\dataset\poems_processed\rhyme\val_d1.jsonl"
   ```

### APPENDIX4: Dataset Processing

1. Conversion from traditional Chinese to simplified Chinese.
2. Extraction of titles and main trunks of poems.
3. Keywords and rhymes extraction then redraft queries using TF-IDF algorithm and jieba package.

Original dataset: 

```
{
  "author": "睿宗皇帝",
  "content": [
   "奇峰嶾嶙箕山北，秀崿岹峣嵩镇南。",
   "地首地肺何曾拟，天目天台倍觉慙。",
   "树影蒙茏鄣叠岫，波深汹涌落悬潭。",
   "唯愿紫宸居得一，永欣丹扆御通三。"
  ],
  "title": "石淙"
 }
```

Training dataset and validation dataset after processed: 五律，五绝，七律，七绝

```
{"query": "请根据这些关键词'长安'，以'安'为韵脚，生成一首以'寄林宽'为主题的五言律诗。五言律诗格式为8小句，每小句字数为5个字，共40个字", "response": "相知四十年，故国与长安。俱喜今辰在，休论往岁难。海鸣秋日黑，山直夏风寒。终始前儒道，升沈尽一般。"}
{"query": "请根据这些关键词'面目'，以'真'为韵脚，生成一首以'人面竹'为主题的五言绝句。五言绝句格式为4小句，每小句字数为5个字，共20个字", "response": "共识此君面，谁知面目真。清风犹立懦，奇节仰先民。"}
{"query": "请根据这些关键词'堪弄'，以'情'为韵脚，生成一首以'寄题甘露寺北轩'为主题的七言律诗。七言律诗格式为8小句，每小句字数为7个字，共56个字", "response": "曾向蓬莱宫里行，北轩阑槛最留情。孤高堪弄桓伊笛，缥缈宜闻子晋笙。天接海门秋水色，烟笼隋苑暮钟声。他年会著荷衣去，不向山僧说姓名。"}
{"query": "请根据这些关键词'馀年'，以'情'为韵脚，生成一首以'集古 重九日'为主题的七言绝句。七言绝句格式为4小句，每小句字数为7个字，共28个字", "response": "二十馀年别帝京，可能朝市污高情。秋光何处堪消日，漫绕东篱嗅落英。"}
```


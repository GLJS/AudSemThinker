# AudSemThinker: Enhancing Audio-Language Models Through Reasoning over Semantics of Sound

Official repository for the AudSemThinker model and AudSem dataset. This repository contains the code and resources for the AudSemThinker project, focusing on creating a large-scale audio-language dataset from YouTube subtitles and training multimodal models for audio semantic understanding.

## Project Links

| Category | Link                                                                  | Description                                      |
|----------|-----------------------------------------------------------------------|--------------------------------------------------|
| Paper    | [arXiv:2505.14142](https://arxiv.org/abs/2505.14142)                  | AudSemThinker research paper                     |
| Datasets | [gijs/audsem-simple](https://huggingface.co/datasets/gijs/audsem-simple) | Simplified AudSem dataset without semantic descriptors |
|          | [gijs/audsem](https://huggingface.co/datasets/gijs/audsem)             | Full AudSem dataset with semantic descriptors    |
| Models   | [gijs/audsemthinker-qa](https://huggingface.co/gijs/audsemthinker-qa)       | AudSemThinker model fine-tuned for QA        |
|          | [gijs/audsemthinker](https://huggingface.co/gijs/audsemthinker)           | General AudSemThinker model                      |
|          | [gijs/audsemthinker-qa-grpo](https://huggingface.co/gijs/audsemthinker-qa-grpo) | AudSemThinker QA model with GRPO optimization |
| Demo     | [gijs/audsemthinker](https://huggingface.co/spaces/gijs/audsemthinker)     | Interactive demo of the AudSemThinker model    |

## Repository Overview

The repository is structured into several directories, each serving a specific purpose in the dataset creation and model training pipeline:

- **filtering/**: Filters raw YouTube subtitle data to extract high-quality sound descriptions.
- **clip_embeddings/**: Converts audio and text data into embeddings for further filtering based on cosine similarity.
- **metadata_evaluation/**: Generates the final dataset by extracting multimodal features (audio, image, video) and combining them.
- **tarify/**: Packages the dataset into WebDataset tar files for efficient data loading.
- **training/**: Contains scripts for training models using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO), along with evaluation scripts.

---

## Getting Started

### Environment Setup

We recommend using a Conda environment with Python 3.10:

```bash
conda create -n audsemthinker python=3.10
conda activate audsemthinker
conda install pip
pip install -r requirements.txt
```

---

## Running the Pipeline

Below are detailed instructions and example commands for each stage of the pipeline.

### 1. Filtering Stage (`filtering/`)

This stage processes raw YouTube subtitles to extract and refine sound event descriptions.

**Download Subtitles:**
```bash
python filtering/download/download_pd.py --output_dir <output_dir> --other_args <values>
```

**Preprocessing:**
```bash
python filtering/preprocessing/main.py --input_dir <input_dir> --output_dir <output_dir>
```

**Classification (BERT-based and Mistral-based):**
```bash
python filtering/classification/main.py --input_dir <input_dir> --output_dir <output_dir>
python filtering/mistral_classification/main.py --input_dir <input_dir> --output_dir <output_dir>
```

**Combining Results:**
```bash
python filtering/02_combining.py --input_dirs <classification_output_dirs> --output_dir <combined_output_dir>
python filtering/04_combination.py --input_dir <combined_output_dir> --output_dir <final_filtered_output_dir>
```

---

### 2. Embedding-based Filtering (`clip_embeddings/`)

Refines the dataset using audio and text embeddings.

```bash
python clip_embeddings/prepreprocess.py --input_dir <input_dir> --output_dir <output_dir>
python clip_embeddings/preprocess_audio.py --input_dir <input_dir> --output_dir <output_dir>
python clip_embeddings/process_audio.py --input_dir <input_dir> --output_dir <output_dir>
python clip_embeddings/process_text.py --input_dir <input_dir> --output_dir <output_dir>
python clip_embeddings/postprocess_sim.py --input_dir <input_dir> --output_dir <output_dir>
python clip_embeddings/postpostprocess.py --input_dir <input_dir> --output_dir <output_dir>
```

---

### 3. Metadata Evaluation (`metadata_evaluation/`)

Extract multimodal features and generate the final dataset.

**Run Feature Extractors:**
```bash
python metadata_evaluation/audio_model/main.py --data_dir <data_dir> --num_workers 16 --batch_size 16 --num_shards 100 --start_shard 0
python metadata_evaluation/image_model/main.py --data_dir <data_dir> --num_workers 16 --batch_size 16 --num_shards 100 --start_shard 0
python metadata_evaluation/video_model/main.py --data_dir <data_dir> --num_workers 16 --batch_size 16 --num_shards 100 --start_shard 0
```

**Combine and Process Metadata:**
Run scripts sequentially from 01 to 05:
```bash
python metadata_evaluation/combine_data/01_combine_outputs.py --input_dir <input_dir> --output_dir <output_dir>
python metadata_evaluation/combine_data/02_postfiltering.py --input_dir <input_dir> --output_dir <output_dir>
python metadata_evaluation/combine_data/03_opensource_caption_creation_batch.py --input_dir <input_dir> --output_dir <output_dir>
python metadata_evaluation/combine_data/04_opensource_qa_multiple_choice_generation_batch.py --input_dir <input_dir> --output_dir <output_dir>
python metadata_evaluation/combine_data/05_get_data_files.py --input_dir <input_dir> --output_dir <output_dir>
```

---

### 4. Tarification (`tarify/`)

Package data into WebDataset tar files:

```bash
python tarify/laion-tarify.py --input_dir <input_dir> --output_dir <output_dir>
```

---

### 5. Data Preparation for Training (`filtering/preparation/`)

Create WebDataset shards for training:

```bash
python 01_create_webdataset.py --jsonl_path <path_to_jsonl> --output_dir <output_webdataset_dir>
python 01_create_webdataset_mc_qa.py --jsonl_path <path_to_mc_qa_jsonl> --output_dir <output_mc_qa_webdataset_dir> --semantic
python 01_create_webdataset_qa.py --jsonl_path <path_to_qa_jsonl> --output_dir <output_qa_webdataset_dir> --semantic
```

---

### 6. Model Training (`training/`)

**Supervised Fine-Tuning (SFT):**
```bash
python main.py --shard_folder <path_to_shards> --no_debug --semantic_elements
```

**Group Relative Policy Optimization (GRPO):**
```bash
# Start VLLM server
CUDA_VISIBLE_DEVICES=3 python -m trl.scripts.vllm_serve --model <MODEL_PATH> --tensor_parallel_size 1 &

# Run GRPO training
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --num_processes 3 --use_deepspeed --zero_stage 3 grpo_main.py --optimization lora --model_id_or_path <MODEL_PATH> --name <GRPO_RUN_NAME> --train_batch_size 2 --num_generations 6 --no_debug --shard_folder <path_to_mc_qa_shards>
```

---

### 7. Evaluation (`training/evaluate/` and `training/AudioBench/`)

**MMAU Evaluation:**
```bash
python mmau_mini.py --model_path <model_checkpoint> --semantic_elements
python mmau_mini_omni_targetlength.py --model_path <model_checkpoint> --target_length 25
```

**AudioBench Evaluation:**
```bash
# Start evaluation server
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model casperhansen/llama-3-70b-instruct-awq --quantization awq --port 5001 &

# Run evaluation
python src/main_evaluate.py --dataset_name <dataset_subset> --model_name audsemthinker --batch_size 8 --metrics llama3_70b_judge --only_generate_predictions
```

### Citation
If you use the AudSemThinker model or AudSem dataset in your research, please cite the accompanying paper:
```
@misc{wijngaard2025audsemthinkerenhancingaudiolanguagemodels,
  title={AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound}, 
  author={Gijs Wijngaard and Elia Formisano and Michele Esposito and Michel Dumontier},
  year={2025},
  eprint={2505.14142},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2505.14142}, 
}
```
# Canadian Parliament Fact-Check

This project builds a fact-checking dataset from Canadian parliamentary debate data and fine-tunes an instruction model to assess political claims. The core data source is a local PostgreSQL database of parliamentary records, including Hansard debate text and structured bill and vote information.

The final practical workflow in this repo is: collect claim candidates from parliamentary text, label them with a mix of deterministic parliamentary facts and LLM-assisted verification, train on the resulting prompt/completion dataset, and export the final merged model to GGUF for lightweight inference workflows.

## Current Workflow

1. Extract or collect claim candidates from Hansard-style parliamentary text and related processed claim files.
2. Build a labeled dataset using:
   - deterministic parliamentary facts from the local PostgreSQL database
   - LLM-assisted labeling for broader natural-language claims
3. Save the final training dataset to [`data/training_pairs.json`](data/training_pairs.json).
4. Fine-tune the base model with the training notebook or the Python training script.
5. Convert the merged Hugging Face model to GGUF for downstream inference.
6. Run GGUF-based inference with the standalone inference script.

## Dataset

The final training file used in this project is [`data/training_pairs.json`](data/training_pairs.json).

Each row contains:
- `prompt`
- `completion`
- `metadata`

The dataset uses these labels:
- `TRUE`
- `FALSE`
- `MISLEADING`
- `UNVERIFIED`

The final dataset combines two main sources of supervision:
- structured parliamentary fact claims such as bill introduction dates, bill status, and party voting records
- broader natural-language claims labeled with LLM assistance

Current dataset summary:
- `1289` training rows
- weighted toward structured claim types such as `status_snapshot`, `introduced_date`, and `party_vote_direction`

## Training

The main training setup uses:
- base model: `meta-llama/Llama-3.1-8B-Instruct`
- training approach: LoRA fine-tuning
- training stack: PyTorch, Transformers, TRL, PEFT, bitsandbytes

Primary training workflow:
- notebook-first training in [`notebooks/factcheck_llama_lora_fullft_colab.ipynb`](notebooks/factcheck_llama_lora_fullft_colab.ipynb)
- optional Python training entrypoint in [`src/train.py`](src/train.py)

At a high level, training uses:
- 4-bit model loading for efficiency
- LoRA adapters for parameter-efficient fine-tuning
- instruction-style prompt/completion supervision from `training_pairs.json`

Core training configuration lives in [`config/model_config.py`](config/model_config.py).

## GGUF Export

After fine-tuning, the merged Hugging Face model is converted to GGUF format for lightweight inference setups.

This export path is documented in:
- [`notebooks/factcheck_gguf_export_colab.ipynb`](notebooks/factcheck_gguf_export_colab.ipynb)

The GGUF conversion step uses `llama.cpp` tooling on the merged fine-tuned model output.

## Inference

The repo also includes a standalone GGUF inference path in [`inference.py`](inference.py).

This script:
- loads a GGUF model with `llama_cpp`
- runs inference over a JSON evaluation file such as `send-to-vince.json`
- writes the generated outputs and metadata to an output JSON file

This is the main lightweight inference path for the current GGUF model flow.

## Models and Tools Used

- Fine-tuning base model: `meta-llama/Llama-3.1-8B-Instruct`
- Labeling support: OpenAI API for LLM-assisted claim labeling
- Training stack: PyTorch, Transformers, TRL, PEFT, bitsandbytes
- Data source: local PostgreSQL parliamentary database
- GGUF export tooling: `llama.cpp`
- GGUF inference runtime: `llama_cpp`

## Repo Pointers

- [`data/training_pairs.json`](data/training_pairs.json): final training dataset
- [`src/`](src): extraction, dataset-building, and training scripts
- [`notebooks/`](notebooks): Colab training and GGUF export workflows
- [`config/model_config.py`](config/model_config.py): core training configuration
- [`inference.py`](inference.py): standalone GGUF inference script

## Code

The repository is organized around the main stages of the project:

- `data/` contains the datasets used for training and evaluation. The most important file here is `training_pairs.json`, which is the final training set.
- `src/` contains the pipeline scripts for claim extraction, dataset building, training preparation, and some older optional inference code.
- `notebooks/` contains the Colab notebooks used for fine-tuning and GGUF export.
- `config/model_config.py` holds the main training configuration.
- `inference.py` is the standalone GGUF inference script used after export.

## Backend and Frontend Notes

Backend work in this repo mainly covers:
- dataset-building scripts
- training scripts and notebooks
- optional inference scaffolding
- standalone GGUF inference with `llama_cpp`

No dedicated frontend application is part of the final workflow documented here.

## Optional / Legacy Notes

Older manual-label and API-serving code still exists in the repo, but it is not the only inference path. The newer standalone GGUF inference script is also part of the current repo workflow.

Docker and API files are kept as optional or earlier-stage infrastructure rather than the primary path for reproducing the final project workflow.

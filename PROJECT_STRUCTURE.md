parliament-factcheck/\
├── README.md                   # Main project documentation\
├── PROJECT_STRUCTURE.md        # This file - project organization guide\
├── pyproject.toml             # Python dependencies (uv package manager)\
├── uv.lock                    # Locked dependency versions\
├── docker-compose.yml         # Docker orchestration for PostgreSQL + API\
├── Dockerfile                 # Main application container\
├── Dockerfile.api             # API service container\
├── activate.sh                # Environment activation script\
├── requirements.txt           # Pip-compatible requirements (optional)\
├── .gitignore                 # Git ignore rules\
│\
├── data/                      # Data storage (gitignored)\
│   ├── extracted_claims.json      # Raw claims from debates\
│   ├── labeled_claims.json        # Manually verified claims\
│   └── training_pairs.json        # Formatted for LLM training\
│\
├── models/                    # Trained model storage (gitignored)\
│   ├── factcheck-llama-canadian/  # Fine-tuned model\
│   └── checkpoints/               # Training checkpoints\
│\
├── src/                       # Source code\
│   ├── init.py\
│   ├── data_loader.py         # Database interface\
│   ├── extract_claims.py      # Claim extraction script\
│   ├── label_claims.py        # CLI labeling tool\
│   ├── create_synthetic_data.py # Synthetic training data generator\
│   ├── labeling_tool.py       # Streamlit GUI for labeling\
│   ├── train.py               # LLM fine-tuning script\
│   ├── api.py                 # FastAPI inference service\
│   └── evidence_retriever.py  # Evidence retrieval system\
│\
├── config/                    # Configuration files\
│   └── model_config.py        # Model training parameters\
│\
├── notebooks/                 # Jupyter notebooks (exploration)\
│\
└── logs/                      # Training and application logs

# ConLLM: Contrastive + LLM-based Deepfake Detection System

## Overview
This project implements **ConLLM**, a state-of-the-art deepfake detection system that leverages contrastive learning and large language models (LLMs) for robust detection across audio, video, and audio-visual modalities. The pipeline is designed for research-grade experiments using full, real-world datasets.

## Datasets
The following datasets are supported (not included, must be placed in the project root):
- **Audio:** ASVSpoof 2019 (LA), DECRO (D-E and D-C)
- **Video:** Celeb-DF (CDF), WildDeepfake (WD)
- **Audio-Visual:** FakeAVCeleb (FAFC), DFDC (DeepFake Detection Challenge)

Each dataset should follow its official directory structure as described in the project documentation.

## Project Structure
```
Deepfake/
├── datasets/                # Dataset loader scripts
├── embeddings/              # Embedding extraction scripts and outputs
├── models/                  # Model training, refinement, evaluation scripts
├── data/                    # (Optional) Raw data location
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── ...
```

## Setup Instructions
1. **Clone the repository and install dependencies:**
   ```bash
   git clone <your-repo-url>
   cd Deepfake
   pip install -r requirements.txt
   ```
2. **Download and organize datasets** as described above.
3. **(Optional) Set up GPU support** for faster training and inference.

## Usage
### 1. Extract Embeddings
Run the scripts in `embeddings/` to extract modality-specific embeddings:
```bash
python embeddings/extract_audio_embeddings.py
python embeddings/extract_video_embeddings.py
python embeddings/extract_av_embeddings.py
```

### 2. Contrastive Learning
Align embeddings in a shared latent space:
```bash
python models/contrastive_learning.py
```

### 3. LLM-Based Refinement
Refine aligned embeddings using the LLM-based module:
```bash
python models/llm_refinement.py
```

### 4. Classification
Train and evaluate the classifier:
```bash
python models/classification.py
```

### 5. Evaluation & Analysis
- **Performance Evaluation:**
  ```bash
  python models/evaluation.py
  ```
- **Ablation Studies:**
  ```bash
  python models/ablation.py
  ```
- **Cross-Lingual Generalization:**
  ```bash
  python models/crosslingual.py
  ```
- **Computational Benchmarking:**
  ```bash
  python models/benchmark.py
  ```

## Experiment Workflow
1. Prepare datasets and extract embeddings for all modalities.
2. Align embeddings using contrastive learning.
3. Refine embeddings with the LLM-based module.
4. Train and evaluate the classifier.
5. Run ablation, cross-lingual, and benchmarking scripts for comprehensive analysis.

## Notes on Performance
- **Performance (speed, memory, FLOPs) will depend on your system hardware** (CPU, GPU, RAM, etc.).
- For best results, use a machine with a modern GPU and sufficient memory.
- Reported metrics (EER, ACC, AUC) may vary based on hardware and dataset size.



**For questions or contributions, please open an issue or pull request.** 
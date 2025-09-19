# Enzyformer
Enzyformer is a **Two-Stage Pretrained Model for Enzymatic Retrosynthesis**.

## Environment Setup

### 1. Enzymatic Retrosynthesis
Refer to the Chemformer environment setup:  
[https://github.com/MolecularAI/Chemformer](https://github.com/MolecularAI/Chemformer)

### 2. Enzyme Assignment
Create a dedicated environment and install required packages:

```bash
# Create a Python 3.10 environment
mamba create -n enzymeformer python=3.10
mamba activate enzymeformer

# Install PyTorch and corresponding CUDA (adjust based on your GPU)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install DGL (Deep Graph Library)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps

# Install rxnfp for reaction fingerprints
pip install rxnfp==0.1.0 --no-deps

# Install Hugging Face Transformers
pip install transformers

# Install Simple Transformers
pip install simpletransformers
```

# Retrosynthesis Prediction

## Data Preparation
1. **Download dataset**  
   [Google Drive Dataset](https://drive.google.com/drive/folders/14rY863a-qdngGUnbF6BB7OJEJ8X6Sv5x?usp=drive_link)  
   Place all files in `./Retrosynthesis/data`.

2. **Download pretrained checkpoints**  
   [Google Drive Checkpoints](https://drive.google.com/drive/folders/1hWeqqLjWYTOrwrvg1P3k7Uj8QyJ5US0o?usp=drive_link)  
   Place all files in `./Retrosynthesis/ckpt`.

---

## Training and Inference

Run the following commands for **training the model if needed** and **direct inference on Enzyformer**:

```bash
cd ./Chemformer

# Train the model (if needed)
bash ../finetune-backward.sh

# Or directly run inference on Enzyformer
bash ../finetune-direct.sh

# Score predictions
python score_predictions.py \
    -mode 2 \
    -beam_size 10 \
    -n_best 10 \
    -augmentation 1 \
    -targets ./retrosyntheis/data/test.targets.txt \
    -process_number 8 \
    -score_alpha 1 \
    -save_file ./retrosyntheis/otuputs/final_results_mode_5_wo.txt \
    -detailed \
    -source ./retrosyntheis/data/test.sources.txt
```
## EC number assignment







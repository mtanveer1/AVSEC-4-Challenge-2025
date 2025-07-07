# AVSEC-4-Challenge-2025
This repository implements **Bicrosswin**, a novel bimodal framework for **Audio-Visual Speech Enhancement (AVSE)**. It leverages the power of **Swin Transformer V2** for visual encoding, a **UNet-based waveform encoder-decoder**, **bi-directional cross-attention fusion**, and **Squeezeformer** for temporal modeling.

## Requirements
..

```bash
pip install -r requirements.txt
```

## Folder Structure

AVSE_2025/
├── code/
│   ├── _pycache_/
│   ├── conf/
│       ├── eval.yaml
│       └── train.yaml
│   ├── outputs/
│   ├── dataset.py
│   ├── model.py
│   ├── test.py
│   ├── train.py
│   └── utils.py
├── data_preparation/
│   ├── avse1/
│   └── avse4/
│       ├── .hydra/
│       ├── clarity/
│       ├── hydra/
│       ├── multirun/
│       ├── build_scenes.log
│       ├── build_scenes.py
│       ├── config.yaml
│       ├── create_speech_masks.py
│       ├── render_scenes.log
│       ├── render_scenes.py
│       └── setup_avsec_data
├── evaluation/
│   ├── avse1/
│       ├── config.yaml
│       └── objective_evaluation.py
│   └── avse4/
│       ├── mbstoi/
│       ├── config.yaml
│       └── objective_evaluation.py

## Configuration
### Training: train.yaml

```bash
data.root: "D:/AVSE_2025/avse_data"
frontend_ckpt_path: "D:/Download/swin_base_patch4_window12_384_22k.pth"
batch_size: 2
sample_rate: 16000
rgb: True
trainer.gpus: 1
trainer.max_epochs: 100
```

Download Swin Transformer V2 weights from: https://github.com/microsoft/Swin-Transformer

### Evaluation: eval.yaml

'''bash
ckpt_path: "D:/AVSE_2025/avse_challenge2/code/logs2/lightning_logs/version_28/checkpoints/model-epoch=18-val_loss=0.006.ckpt"
save_dir: "D:/AVSE_2025/sample_data/eval/scene_penult"
model_uid: "mono_"
eval_set: True
cpu: False

## Usage
### Train

```bash
python train_script.py \
  --log_dir D:/AVSE_2025/logs \
  --batch_size 2 \
  --lr 0.0001 \
  --gpu 1 \
  --max_epochs 100
```
### Evaluate

```bash
python test.py \
  --ckpt_path D:/AVSE_2025/avse_challenge2/code/logs2/lightning_logs/version_28/checkpoints/model-epoch=18-val_loss=0.006.ckpt \
  --save_root D:/AVSE_2025/sample_data/eval/scene_penult \
  --model_uid mono_ \
  --dev_set False \
  --eval_set True \
  --cpu False
  ```

## Objective Metrics
```bash
python objective_evaluation.py
```

Make sure your config.yaml has correct paths for:
```bash
target: D:/AVSE_2025/avse_data/dev/scenes
enhanced: D:/AVSE_2025/avse_data/eval/scenes/default_model
```

AVSE Challenge Dataset: https://github.com/cogmhear/avse_challenge/tree/main/data_preparation/avse4

## Contact Us










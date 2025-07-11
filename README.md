# AVSEC-4-Challenge-2025
## Requirements
..
You can install all requirements using 
```bash
pip install -r requirements.txt
```

## Usage

```bash
#Expected folder structure
avsec4/
├── train/
│   └── scenes/
│       ├── S34526_mix.wav
│       ├── S34526_mono_mix.wav
│       ├── S34526_interferer.wav
│       ├── S34526_interferer_mono.wav
│       ├── S34526_target.wav
│       ├── S34526_target_anechoic.wav
│       ├── S34526_target_mono.wav
│       ├── S34526_target_mono_anechoic.wav
│       └── S34526_silent.mp4
├── dev/
│   └── scenes/
│       ├── S37890_mix.wav
│       ├── S37890_mono_mix.wav
│       ├── S37890_interferer.wav
│       ├── S37890_interferer_mono.wav
│       ├── S37890_target.wav
│       ├── S37890_target_anechoic.wav
│       ├── S37890_target_mono.wav
│       ├── S37890_target_mono_anechoic.wav
│       └── S37890_silent.mp4
└── eval/
    └── scenes/
        ├── S34526_mix.wav
        ├── S34526_mono_mix.wav
        └── S34526_silent.mp4

```

## Dataset

Please find the dataset details here: [https://github.com/cogmhear/avse_challenge/tree/main/data_preparation/avse4](https://github.com/cogmhear/avse_challenge)

Download Swin Transformer V2 weights from: [https://github.com/microsoft/Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

## Train

Update data_root and frontend_ckpt_path in [main/conf/train.yaml](https://github.com/mtanveer1/AVSEC-4-Challenge-2025/blob/main/conf/train.yaml)

```bash
python train.py data.root="./avsec4" data.num_channels=1 trainer.log_dir="./logs" data.batch_size=8 trainer.accelerator=gpu trainer.gpus=1

more arguments in conf/train.yaml
```

## Test

Update data_root, ckpt_path and save_dirin [main/conf/eval.yaml](https://github.com/mtanveer1/AVSEC-4-Challenge-2025/blob/main/conf/eval.yaml)

```bash
usage: test.py [-h] 
               --ckpt_path ./model.pth 
               --save_dir ./enhanced 
               --model_uid avse 
               [--dev_set False] 
               [--eval_set True] 
               [--cpu False]

# Model evaluation on the dev set
python test.py \
  --model_uid mono_ \
  --dev_set True \
  --eval_set False \
  --cpu False

# Model evaluation on the eval set
python test.py \
  --model_uid mono_ \
  --dev_set False \
  --eval_set True \
  --cpu False
```

## Evaluation

Update data_root:  [main/evaluation/config.yaml](https://github.com/mtanveer1/AVSEC-4-Challenge-2025/blob/main/evaluation/config.yaml)

```bash  
python objective_evaluation.py
```

## Contact Us
For further queries, contact us at:

[Deepanshu Gupta](https://github.com/Deepanshu41008) - mc230041008@iiti.ac.in

[Harshith Ganji](https://github.com/Aach1) - harshithjaiganji@gmail.com

[Yash Modi](https://github.com/YashModi21) - yashmodi017@gmail.com

[Sanskriti Jain](https://github.com/Sanskriti-hello) - sansjain23.11@gmail.com









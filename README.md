# AVSEC-4-Challenge-2025

# AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement

Kindly cite this paper if you are using this code: 

Sajid, M., Deepanshu Gupta, Yash Modi, Sanskriti Jain, Harshith Jai Surya Ganji, A. Rahaman, Harshvardhan Choudhary, Nasir Saleem, Amir Hussain, and M. Tanveer (2025). "AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement, 4th Cogmhear Audio-Visual Speech Enhancement Challenge (AVSEC), Interspeech 2025"

Paper Link: https://doi.org/10.21437/AVSEC.2025-13

Arxiv Link: https://doi.org/10.48550/arXiv.2510.05295

BibTex
------
```
@inproceedings{sajid25_avsec,
  title     = {{AUREXA-SE: Audio-Visual Unified Representation Exchange Architecture with Cross-Attention and Squeezeformer for Speech Enhancement}},
  author    = {M. Sajid and Deepanshu Gupta and Yash Modi and Sanskriti Jain and Harshith Jai Surya Ganji and A. Rahaman and Harshvardhan Choudhary and Nasir Saleem and Amir Hussain and M Tanveer},
  year      = {2025},
  booktitle = {{4th Cogmhear Audio-Visual Speech Enhancement Challenge (AVSEC)}},
  pages     = {56--61},
  doi       = {10.21437/AVSEC.2025-13},
}
```

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
## Pretrained Model Checkpoints

The pretrained checkpoints used in the paper can be downloaded from:

**Google Drive:**  
[Click Here](https://drive.google.com/file/d/15QNLvn_v4yLzOJAxh9FNqKHk5rOKq27g/view?usp=sharing)

## Contact Us
For further queries, contact us at:

[Deepanshu Gupta](https://github.com/Deepanshu41008) - mc230041008@iiti.ac.in

[Harshith Ganji](https://github.com/Aach1) - harshithjaiganji@gmail.com

[Yash Modi](https://github.com/YashModi21) - yashmodi017@gmail.com

[Sanskriti Jain](https://github.com/Sanskriti-hello) - sansjain23.11@gmail.com

[M. Sajid](https://github.com/MdSajid1044) - sajid.mathml@gmail.com









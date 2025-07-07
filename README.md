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
│   └── avse4/
│       ├── mbstoi/
│       ├── config.yaml
│       └── objective_evaluation.py
```

## Dataset

Please find the dataset details here: https://github.com/cogmhear/avse_challenge/tree/main/data_preparation/avse4

Download Swin Transformer V2 weights from: https://github.com/microsoft/Swin-Transformer

## Train

```bash
python train_script.py \
  --log_dir D:/AVSE_2025/logs \
  --batch_size 2 \
  --lr 0.0001 \
  --gpu 1 \
  --max_epochs 100
```

## Test

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
  --ckpt_path D:/AVSE_2025/checkpoints/model-epoch=18-val_loss=0.006.ckpt \
  --save_dir D:/AVSE_2025/sample_data/eval/scene_dev \
  --model_uid mono_ \
  --dev_set True \
  --eval_set False \
  --cpu False

# Model evaluation on the eval set
python test.py \
  --ckpt_path D:/AVSE_2025/checkpoints/model-epoch=18-val_loss=0.006.ckpt \
  --save_dir D:/AVSE_2025/sample_data/eval/scene_eval \
  --model_uid mono_ \
  --dev_set False \
  --eval_set True \
  --cpu False
```

## Evaluation

```bash  
python objective_evaluation.py
```

## Contact Us
For further queries, contact us at:

[Deepanshu Gupta](https://github.com/Deepanshu41008) - mc230041008@iiti.ac.in

[Harshith Ganji](https://github.com/Aach1) - harshithjaiganji@gmail.com

[Yash Modi](https://github.com/YashModi21)

[Sanskriti Jain](https://github.com/Sanskriti-hello) - sansjain23.11@gmail.com









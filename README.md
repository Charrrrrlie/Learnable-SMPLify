# Learnable SMPLify: A Neural Solution for Optimization-Free Human Pose Inverse Kinematics

#### <p align="center">[arXiv Paper](https://arxiv.org/abs/2508.13562) | [Hugging Face](https://huggingface.co/Charlie019/Learnable-SMPLify)</p>


![framework](assets/framework.png)

``TL;DR`` Given X_{t-s} and X_{t} 3D keypoints, 
calculate residual SMPL parameters from t-s to t.

## Preparation
Refer to [PREPARATION.md](doc/PREPARATION.md) for installation and data preparation details.

## Checkpoints
The pretrained model checkpoint is available at [Google Drive](https://drive.google.com/drive/folders/1oyG2gbB3EMcc6NgTIT1p1uJ_Em0dJwXz?usp=sharing) and [Hugging Face](https://huggingface.co/Charlie019/Learnable-SMPLify/blob/main/best_ckpt.pth.tar).

## Usage
### Training
cd to `src` folder and run the following command.

```
torchrun --nproc-per-node <NUM_GPUS> main.py --config configs/net.yaml (--extra_tag <EXTRA_TAG> --batch_size <BATCH_SIZE> --epochs <EPOCHS>)
```

You can get logs, tensorboard and checkpoints in the corresponding `logs/<MODEL_NAME>_net_<EXTRA_TAG>` folder.

### Evaluation
To evaluate the model, run the following command:

```
torchrun --nproc-per-node <NUM_GPUS> main.py --config configs/net.yaml --eval --checkpoint <PATH_TO_CHECKPOINT>
```

### Sequential Inference
To run sequential inference, you can use the following command:

```
python inference.py <PATH_TO_CHECKPOINT> (<DATASET_NAME> <SAMPLE_RATIO>)
```

## Citation
If you find this work useful in your research, please consider citing:

```
@misc{LearnableSMPLify,
      title={Learnable SMPLify: A Neural Solution for Optimization-Free Human Pose Inverse Kinematics},
      author={Yuchen, Yang and Linfeng, Dong and Wei, Wang and Zhihang, Zhong and Xiao, Sun},
      year={2025},
      eprint={2508.13562},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
We thank the authors of [ST-GCN](https://github.com/open-mmlab/mmskeleton), [ReFit](https://github.com/yufu-wang/ReFit), [OSX](https://github.com/IDEA-Research/OSX/tree/main) for their great works. We partially refer to their codebases for this project.
# Exploiting Visual Context Semantics for Sound Source Localization

Xinchi Zhou, Dongzhan Zhou, Hang Zhou, Di Hu, Wanli Ouyang

## Introduction

In this work, we propose a visual reasoning module to explicitly exploit the rich visual context semantics in the sound source localization task. We carefully design the learning objectives to guide the extracted visual semantics and enhance the cross-modal interactions, leading to more robust feature representations and higher localization accuracy.

## Environment

```shell
* Python 3.7.4
* Pytorch 1.9.0
* torchvision 0.10.0
```

You should also install lmdb, opencv-python, pillow, librosa, scipy, and sklearn before you start.

## Dataset Preparation

We conduct experiments on Flickr-SoundNet and VGGSound. Please refer to [Learning to localize sound sources](https://github.com/ardasnck/learning_to_localize_sound_source) and [Localizing Visual Sounds the Hard Way](https://github.com/hche11/Localizing-Visual-Sounds-the-Hard-Way) to download the Flickr-SoundNet and VGGSound data, respectively.

We use the ffmpeg toolbox to convert the raw video clips into audio tracks and a sequence of frames. We extract one frame per second by default. The data should be placed in the following structure.

```
data path
│
└───video_frames
|   └───video_id1
│   │   001.jpg
│   │   002.jpg
│   │
|   └───video_id2
│   │   001.jpg
│   │   002.jpg
│   │
└───audios
    └───video_id1.wav
    └───video_id2.wav
```

Two types of data loader are implemented to fetch the auditory and visual inputs from the dataset. The `LmdbDataSet` will load the whole dataset into RAM to reduce the I/O burdens. To use this dataloader, you need to run the `preprocess_data.py` first to convert the dataset into the LMDB format. If your RAM cannot support loading all data, you can choose the `AVDataset` implementation. Please refer to `preprocess_data.py` for details.

## Usage

We use yaml to manage the hyper-parameters in this repo. Please refer to the `configs` folder. You should revise these configs to provide the right paths to data. You are also encouraged to set an appropriate prefix and name to distinguish experiments (i.e., the exp_prefix and exp_name variable in the config).

```shell
python train.py
```

For evaluation, the path to the pre-trained model should be provided.

```shell
python evaluate.py
```

## Citation

If you find this repo useful, please consider to cite

```
@inproceedings{zhou2023exploiting,
  title={Exploiting Visual Context Semantics for Sound Source Localization},
  author={Zhou, Xinchi and Zhou, Dongzhan and Hu, Di and Zhou, Hang and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={5199--5208},
  year={2023}
}
```

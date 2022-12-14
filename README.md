# [moai-challenge](./.github/inference.pdf)
> 링크를 통해 챌린지 수행 과정에 대한 구체적인 내용을 확인할 수 있습니다.

Chest CT에서의 인공지능 기반 자동 Body morphometry 측정에 대한 구조를 포함하고 있습니다.

## Team members
- sangho Kim | [[github]](https://github.com/sangh0)
- yujin Kim | [[github]](https://github.com/yujinkim1)

## Requirements
- [Python](https://www.python.org/downloads) 3.9 or above
- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [CuDNN](https://developer.nvidia.com/cudnn)
- [Pytorch](https://pytorch.org/docs/stable/index.html)

## [Team Convention](./.github/CONVENTION.md)
> 링크를 통해 팀의 컨벤션을 확인할 수 있습니다.

## Architecture
### [U-Net(MICCAI 2015) Paper](https://arxiv.org/pdf/1505.04597.pdf)
<img src="https://miro.medium.com/max/1200/1*qNdglJ1ORP3Gq77MmBLhHQ.png" />

## Prepare and activate environment on anaconda
```bash
$ conda env create --file environment.yaml
$ conda activate moai2022
```

## Run dcm to png format converter
```bash
$ sudo python3 ./dcm2png_converter.py --original_path {DCM_DIR} --convert_path {./ + 'image'}
```

## Training
```bash
$ sudo python3 ./main.py --weight_save_folder_dir ./{DIR_NAME}--data_dir {DATASET_DIR} --num_classes {INT} --lr {1E} --end_lr {1E} --optimizer {OPTIMIZER} --epochs {INT} --ohem_loss_weight {DOUBLE}
--dice_loss_weight {DOUBLE} --batch_size {INT} --weight_decay {1E} --num_filters {INT}
```

## Testing
```bash
$ sudo python3 ./test.py --data_dir {DATASET_DIR} --weight_dir {WEIGHT_DIR} --submission_dir {SUB_DIR} --submission_save_dir {SAVE_DIR}
```
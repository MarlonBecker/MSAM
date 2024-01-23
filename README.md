# Momentum-SAM: Sharpness Aware Minimization without Computational Overhead

Official implementation of [“Momentum-SAM: Sharpness Aware Minimization without Computational Overhead”](https://arxiv.org/abs/2401.12033).

## How to Use

### Import Optimizer to your code

Simply import the optimizer to your code
```
from optimizer.msam import MSAM
from optimizer.adamW_msam import AdamW_MSAM
```
and use it as a drop-in replacement for SGD or AdamW. If you are not decaying $\rho$ during your training, you should call ```optimizer.move_back_from_momentumAscent()``` at the end of your training to recover unperturbed parameters (see [main.py](https://github.com/MarlonBecker/MSAM/blob/main/main.py)).


### Run Examples

Baselines:

```
python -m torch.distributed.run main.py --logSubDir CIFAR_WRN_baseline --ifile configs/CIFAR100_WRN16_4.ini 
python -m torch.distributed.run main.py --logSubDir CIFAR_ResNet_baseline --ifile configs/CIFAR100_ResNet50.ini 
python -m torch.distributed.run main.py --logSubDir ImageNet_ResNet_baseline --ifile configs/ImageNet_ResNet50.ini 
python -m torch.distributed.run main.py --logSubDir ImageNet_ViT_baseline --ifile configs/ImageNet_ViT.ini 
```

SAM[1]:

```
python -m torch.distributed.run main.py --logSubDir CIFAR_WRN_SAM --ifile configs/CIFAR100_WRN16_4.ini --optimizer SAM --rho 0.2
python -m torch.distributed.run main.py --logSubDir CIFAR_ResNet_SAM --ifile configs/CIFAR100_ResNet50.ini --optimizer SAM --rho 0.2
python -m torch.distributed.run main.py --logSubDir ImageNet_ResNet_SAM --ifile configs/ImageNet_ResNet50.ini --optimizer SAM --rho 0.2
python -m torch.distributed.run main.py --logSubDir ImageNet_ViT_SAM --ifile configs/ImageNet_ViT.ini --optimizer AdamW_SAM --rho 0.2
```

MSAM:

```
python -m torch.distributed.run main.py --logSubDir CIFAR_WRN_MSAM --ifile configs/CIFAR100_WRN16_4.ini --optimizer MSAM --rho 3
python -m torch.distributed.run main.py --logSubDir CIFAR_ResNet_MSAM --ifile configs/CIFAR100_ResNet50.ini --optimizer MSAM --rho 3
python -m torch.distributed.run main.py --logSubDir ImageNet_ResNet_MSAM --ifile configs/ImageNet_ResNet50.ini --optimizer MSAM --rho 3
python -m torch.distributed.run main.py --logSubDir ImageNet_ViT_MSAM --ifile configs/ImageNet_ViT.ini --optimizer AdamW_MSAM --rho 3
```

Additional supported optimizers: ```ESAM```[2],```lookSAM```[3]

# References

[1] Foret et al. 2021 [“Sharpness-Aware Minimization for Efficiently Improving Generalization”](https://openreview.net/forum?id=6Tm1mposlrM)<br>
[2] Du et al. 2022 [“Efficient Sharpness-Aware Minimization for Improved Training of Neural Networks”](https://openreview.net/pdf?id=n0OeTdNRG0Q)<br>
[3] Liu et al. 2020 [“Towards Efficient and Scalable Sharpness-Aware Minimization”](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Towards_Efficient_and_Scalable_Sharpness-Aware_Minimization_CVPR_2022_paper.pdf)

#!/bin/bash


cd /home/anazeri/Transformer_ObjDet_Robustness/

python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.03 --attack_type pgd --save_images False --dataset coco  > pgd_003_coco_resiz_detrR50.txt

python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.05 --attack_type pgd --save_images False --dataset coco > pgd_005_coco_resiz_detrR50.txt

python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.1 --attack_type pgd --save_images False --dataset coco > pgd_01_coco_resiz_detrR50.txt

python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.2 --attack_type pgd --save_images False --dataset coco > pgd_02_coco_resiz_detrR50.txt
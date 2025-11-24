#!/bin/bash


cd /home/anazeri/Transformer_ObjDet_Robustness/



# ## PGD trained on Detr-R50 and kitti:

# python attack_Main.py --backbone resnet50 --resume  /scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/  --epsilon 0.03 --attack_type pgd --save_images True --dataset kitti --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r50_pgd003/ > kitti_pgd_003_origsize_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume  /scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/  --epsilon 0.1 --attack_type pgd --save_images True --dataset kitti --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r50_pgd01/ > kitti_pgd_01_origsize_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume  /scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/  --epsilon 0.03 --attack_type pgd --save_images False --dataset kitti > kitti_pgd_003_resize_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume  /scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/  --epsilon 0.1 --attack_type pgd --save_images False --dataset kitti > kitti_pgd_01_resize_detrR50.txt


# ## PGD trained on Detr-R50-dc5 and kitti:

# python attack_Main.py --backbone resnet50 --resume  /home/anazeri/detr_finetune/outputs12022023_R50DC5_kitti_img6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/ --dilation --epsilon 0.03 --attack_type pgd --save_images True --dataset kitti --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r50dc5_pgd003/ > kitti_pgd_003_origsize_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume  /home/anazeri/detr_finetune/outputs12022023_R50DC5_kitti_img6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/ --dilation --epsilon 0.1 --attack_type pgd --save_images True --dataset kitti --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r50dc5_pgd01/ > kitti_pgd_01_origsize_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume  /home/anazeri/detr_finetune/outputs12022023_R50DC5_kitti_img6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/ --dilation --epsilon 0.03 --attack_type pgd --save_images False --dataset kitti > kitti_pgd_003_resize_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume  /home/anazeri/detr_finetune/outputs12022023_R50DC5_kitti_img6481_epch25/checkpoint.pth --coco_path /home/anazeri/fiftyone/kitti_coco/kitti_val/ --dilation --epsilon 0.1 --attack_type pgd --save_images False --dataset kitti > kitti_pgd_01_resize_detrR50dc5.txt



## PGD trained on Detr-R101 and kitti:

model='/scratch/anazeri/outputs11252023_R101_kitti_img6481_epch25/checkpoint.pth'
path='/home/anazeri/fiftyone/kitti_coco/kitti_val/'
backbonee='resnet101'
attack_type='pgd'
dataset_name='kitti'

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r101_pgd003/ > kitti_pgd_003_origsize_detrR101.txt

python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r101_pdg01/ > pgd_01_kitti_origsize_detrR101.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_003_kitti_resiz_detrR101.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_01_kitti_resiz_detrR101.txt


# ## PGD trained on Detr-R101-dc5 and kitti:

# model='/scratch/anazeri/outputs11272023_R101DC5_kitti_img6481_epch25/checkpoint.pth'
# path='/home/anazeri/fiftyone/kitti_coco/kitti_val/'
# backbonee='resnet101'
# attack_type='pgd'
# dataset_name='kitti'

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r101dc5_gpd003/ > pgd_003_kitti_origsize_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/kitti/kitti_origsize_detr_r101dc5_gpd01/ > pgd_01_kitti_origsize_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_003_kitti_resiz_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_01_kitti_resiz_detrR101dc5.txt


## CW trained on Detr-R50 and kitti:

# model='/scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth'     # model= :true,  model = :wrong. SPACE matters
# path='/home/anazeri/fiftyone/kitti_coco/kitti_val/'
# backbonee='resnet50'
# attack_type='cw'
# dataset_name='kitti'

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --attack_type $attack_type --cw_c 1 --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/CW/kitti/kitti_origsize_detr_r50_cw_c1/ > /home/anazeri/Transformer_ObjDet_Robustness/outputs/kitti/cw_attack/cw_c1_kitti_origsize_detrR50.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --attack_type $attack_type --cw_c 10 --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/CW/kitti/kitti_origsize_detr_r50_cw_c10/ > /home/anazeri/Transformer_ObjDet_Robustness/outputs/kitti/cw_attack/cw_c10_kitti_origsize_detrR50.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --attack_type $attack_type --cw_c 1 --save_images False --dataset $dataset_name > /home/anazeri/Transformer_ObjDet_Robustness/outputs/kitti/cw_attack/cw_c1_kitti_resiz_detrR50.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --attack_type $attack_type --cw_c 10 --save_images False --dataset $dataset_name > /home/anazeri/Transformer_ObjDet_Robustness/outputs/kitti/cw_attack/cw_c10_kitti_resiz_detrR50.txt












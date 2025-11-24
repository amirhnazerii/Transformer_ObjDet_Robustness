#!/bin/bash


cd /home/anazeri/Transformer_ObjDet_Robustness/








## PGD trained on Detr-R50 and coco:


# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.03 --attack_type fgsm --save_images False --dataset coco --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r50_adv003/ #> pgd_003_coco_origsize_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.1 --attack_type pgd --save_images True --dataset coco --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r50_adv01/ > pgd_01_coco_origsize_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.03 --attack_type pgd --save_images False --dataset coco > pgd_003_coco_resiz_detrR50.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/  --epsilon 0.1 --attack_type pgd --save_images False --dataset coco > pgd_01_coco_resiz_detrR50.txt




## PGD trained on Detr-R50-dc5 and coco:


# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/ --dilation  --epsilon 0.03 --attack_type pgd --save_images True --dataset coco --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r50dc5_adv003/ > pgd_003_coco_origsize_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/ --dilation  --epsilon 0.1 --attack_type pgd --save_images True --dataset coco --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r50dc5_adv01/ > pgd_01_coco_origsize_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/ --dilation  --epsilon 0.03 --attack_type pgd --save_images False --dataset coco > pgd_003_coco_resiz_detrR50dc5.txt

# python attack_Main.py --backbone resnet50 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth --coco_path /home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/ --dilation --epsilon 0.1 --attack_type pgd --save_images False --dataset coco > pgd_01_coco_resiz_detrR50dc5.txt



# ## PGD trained on Detr-R101 and coco:


# dataset_name='coco'
# path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
# model_name_ls=('r50' 'r50dc5' 'r101' 'r101dc5')
# model=('https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth')
# backbone_ls=('resent50' 'resnet101')
# attack_type='PGD'
# epsilon_list=(0.03 0.1)
# epsilon_str_list=('003' '01')





# dataset_name='coco'
# path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
# model_name_ls=('r50')
# model=('https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth')
# backbone_ls=('resnet50')
# attack='fgsm'
# epsilon_list=(0.1)
# epsilon_str_list=('01')

# for i in ${!model_name_ls[@]}; do
#   for j in ${!epsilon_str_list[@]}; do
  
# #       if [[ $model_name_ls[$i] = *'dc5'* ]]; then  #check string contains this substring.

# #           python attack_Main.py --backbone $backbone_ls --resume $model --dilation --coco_path $path --epsilon $epsilon_list --attack_type $attack --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/$attack_type/$dataset_name/${dataset_name}_origsize_detr_${model_name_ls[$i]}_pgd${epsilon_str_list[$j]}/ > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack_type}/${dataset_name}/${attack_type}_${epsilon_str_list[$j]}_${dataset_name}_detr${model_name_ls[$i]}.txt
# #       else
# #           python attack_Main_v2.py --backbone $backbone_ls --resume $model --coco_path $path --epsilon $epsilon_list --attack_type $attack --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/$attack_type/$dataset_name/${dataset_name}_origsize_detr_${model_name_ls[$i]}_pgd${epsilon_str_list[$j]}/ > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack_type}/${dataset_name}/${attack_type}_${epsilon_str_list[$j]}_${dataset_name}_detr${model_name_ls[$i]}.txt



#           python attack_Main_v2.py --backbone $backbone_ls --resume $model --coco_path $path  --attack yes --attack_type $attack --epsilon $epsilon_list --save_images_tensor True --dataset $dataset_name --save_images_tensor_path /scratch/anazeri/$attack/$dataset_name/${dataset_name}_resiz_imgTens_detr_${model_name_ls[$i]}_$attack_${epsilon_str_list[$j]}/ > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}/${attack}_${epsilon_str_list[$j]}_${dataset_name}_resiz_tens_detr${model_name_ls[$i]}.txt

# #       fi

#     done
# done




# python attack_Main.py --backbone $backbone_ls[$i] --resume $model --coco_path $path --epsilon $epsilon_list[$i] --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r101_gpd003/ > pgd_003_coco_origsize_detrR101.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r101_pdg01/ > pgd_01_coco_origsize_detrR101.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_003_coco_resiz_detrR101.txt

# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_01_coco_resiz_detrR101.txt


# ## PGD trained on Detr-R101-dc5 and coco:

# model='https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth'     # model= :true,  model = :wrong. SPACE matters
# path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
# backbonee='resnet101'
# attack_type='pgd'
# dataset_name='coco'

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r101dc5_gpd003/ > pgd_003_coco_origsize_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/PGD/coco/coco_origsize_detr_r101dc5_pdg01/ > pgd_01_coco_origsize_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.03 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_003_coco_resiz_detrR101dc5.txt

# python attack_Main.py --backbone $backbonee --resume $model --dilation  --coco_path $path --epsilon 0.1 --attack_type $attack_type --save_images False --dataset $dataset_name > pgd_01_coco_resiz_detrR101dc5.txt



## CW trained on Detr-R50 and coco:

# model='https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'     # model= :true,  model = :wrong. SPACE matters
# path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
# backbonee='resnet50'
# attack_type='cw'
# dataset_name='coco'


dataset_name='coco'
# path='/home/anazeri/fiftyone/kitti_coco/kitti_val/'
path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
model_name_ls=('r50')
# model=('/scratch/anazeri/outputs11202023_R50_kitti_imgs6481_epch25/checkpoint.pth')
model=('https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth')     # model= :true,  model = :wrong. SPACE matters
backbone_ls=('resnet50')
attack='cw'
cw_c_val=(3)
cw_c_str_list=('cw3')



for i in ${!model_name_ls[@]}; do
  for j in ${!model_name_ls[@]}; do
  
    echo ${model_name_ls[$i]}
    if [[ ${model_name_ls[$i]} = *'dc5'* ]]; then  #check string contains this substring.
        echo dilation activated 
        python attack_Main_v2.py --backbone ${backbone_ls[$i]} --resume ${model[$i]} --dilation --coco_path $path  --attack no --attack_type $attack --epsilon $cw_c_val --save_images_tensor False --dataset $dataset_name --save_images_tensor_path /scratch/anazeri/$attack/$dataset_name/${dataset_name}_resiz_imgTens_detr_${model_name_ls[$j]}_${attack}_${cw_c_str_list}/ > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}/${attack}_${cw_c_str_list}_${dataset_name}_resiz_tens_detr${model_name_ls[$j]}to_detr${model_name_ls[$i]}.txt
          
    else
        python attack_Main_v2.py --backbone ${backbone_ls[$i]} --resume ${model[$i]} --coco_path $path  --attack no --attack_type $attack --epsilon $cw_c_val --save_images_tensor False --dataset $dataset_name --save_images_tensor_path /scratch/anazeri/$attack/$dataset_name/${dataset_name}_resiz_imgTens_detr_${model_name_ls[$j]}_${attack}_${cw_c_str_list}/ > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}/${attack}_${cw_c_str_list}_${dataset_name}_resiz_tens_detr${model_name_ls[$j]}to_detr${model_name_ls[$i]}.txt
        
    fi
    
    done
done


# python attack_Main.py --backbone $backbonee --resume $model --coco_path $path --attack_type $attack_type --cw_c 1 --save_images True --dataset $dataset_name --save_images_path /scratch/anazeri/CW/coco/coco_origsize_detr_r50_cw_c1/ > /home/anazeri/Transformer_ObjDet_Robustness/outputs/coco/cw_attack/cw_c1_coco_origsize_detrR50.txt










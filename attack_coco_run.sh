#!/bin/bash
#SBATCH --job-name=cDgR101dc5
#SBATCH --nodes=1                    # Keep it on one node
#SBATCH --tasks-per-node=1           # Using 1 process is sufficient for inference/attacks
#SBATCH --cpus-per-task=2            # Allocate 4 CPU cores per process
#SBATCH --mem=80gb                   # Set 130GB memory (adjust if needed)
#SBATCH --time=8:40:00               # 8 hours; adjust based on expected execution time
#SBATCH --gpus-per-node=a100:1       # Request 1 A100 GPU
#SBATCH --output=dag_eps05_step_size_01_untarget_coco_detr_r101dc5_%j.out
#SBATCH --error=dag_eps05_step_size_01_untarget_coco_detr_r101dc5_%j.err


# ============================================================================
# DAG (Dense Adversary Generation) Attack - KITTI Dataset
# Parameters: eps=8/255, step_size=1/255, iters=100, untargeted
# Matches COCO configuration for cross-dataset consistency
# ============================================================================
cd /home/anazeri/Transformer_ObjDet_Robustness/


# Load the required modules
module add anaconda3/2023.09-0
module load cuda/11.8.0


# Activate your conda environment (replace 'Detr_env1' with your environment name)
source activate Detr_env1

dataset_name='coco'
num_classes=91
path='/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'
model_name_ls=('r101dc5')
# model_name_ls=('r50' 'r50dc5' 'r101' 'r101dc5')
# model=('https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth' 'https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth')
# backbone_ls=('resnet50' 'resnet50' 'resnet101' 'resnet101')
model=('https://dl.fbaipublicfiles.com/detr/detr-r101-dc5-a2e86def.pth')
backbone_ls=('resnet101')
attack='dag'
dag_eps_list=(0.5)       # 
dag_eps_str_list=('eps05')
dag_step_size=0.1        # 
dag_iters=100


# Output directory
output_dir="/scratch/anazeri/$attack/$dataset_name/${dataset_name}_origsiz_SavImg_detr_${model_name_ls}_${attack}_eps05_${dag_iters}_stepsize${dag_step_size}_untargeted/"

# Create output directory if it doesn't exist
mkdir -p $output_dir

mkdir -p /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}

for i in ${!model_name_ls[@]}; do
  for j in ${!dag_eps_str_list[@]}; do
    
    dilation_flag=""
    if [[ ${model_name_ls[$i]} = *'dc5'* ]]; then
      dilation_flag="--dilation"
    fi
    
    echo "Running DAG attack on DETR-${model_name_ls[$i]} (${dataset_name}) with eps=${dag_eps_list[$j]}"
    

    python attack_Main_v3_modified.py \
      --backbone ${backbone_ls[$i]} \
      --resume ${model[$i]} \
      --num_classes $num_classes\
      $dilation_flag \
      --coco_path $path \
      --attack yes \
      --attack_type $attack \
      --dag_eps ${dag_eps_list[$j]} \
      --dag_step_size $dag_step_size \
      --dag_iters $dag_iters \
      --save_images True \
      --dataset_file $dataset_name \
      --save_images_path $output_dir
      > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}/${attack}_${dag_eps_str_list[$j]}_${dataset_name}_detr${model_name_ls[$i]}.txt 
      
  done
done












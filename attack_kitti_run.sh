#!/bin/bash
#SBATCH --job-name=r101Rs5
#SBATCH --nodes=1                    # Keep it on one node
#SBATCH --tasks-per-node=1           # Using 1 process is sufficient for inference/attacks
#SBATCH --cpus-per-task=2            # Allocate 4 CPU cores per process
#SBATCH --mem=70gb                   # Set 130GB memory (adjust if needed)
#SBATCH --time=14:40:00               # 8 hours; adjust based on expected execution time
#SBATCH --gpus-per-node=v100:1       # Request 1 A100 GPU
#SBATCH --output=autopgd_eps06_it50_rs5_kitti_detr_r101_%j.out
#SBATCH --error=autopgd_eps06_it50_rs5_kitti_detr_r101_%j.err


# ============================================================================
# AutoPGD Attack - KITTI Dataset
# Parameters:
# This script runs the AutoPGD adversarial attack for DETR on KITTI validation data.
# ============================================================================
cd /home/anazeri/Transformer_ObjDet_Robustness/


# Load the required modules
module add anaconda3/2023.09-0
module load cuda/11.8.0


# Activate your conda environment (replace 'Detr_env1' with your environment name)
source activate Detr_env1


dataset_name='kitti'
num_classes=9

path='/scratch/anazeri/kitti_coco_format/kitti_val'
model_name_ls=('r50')
model=('/scratch/anazeri/detr_finetune/output/detr-r50-KITTI-orighead92fc-100epch/checkpoint0095.pth')
backbone_ls=('resnet50')

attack='autopgd'
# Define epsilon for L∞ perturbation (8/255 ~ 0.0313725)
pgd_eps_list=(0.6)
pgd_eps_str_list=('eps0.6')
# Number of gradient steps for AutoPGD
pgd_iters=50
# Number of random restarts for AutoPGD
autopgd_restarts=5

# Output directory
output_dir="/scratch/anazeri/${attack}/${dataset_name}/${dataset_name}_origsiz_SavImg_detr_${model_name_ls}_${attack}_${pgd_eps_str_list}_${pgd_iters}it_${autopgd_restarts}rs/"

# Create output directory if it doesn't exist
mkdir -p $output_dir

mkdir -p /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}

for i in ${!model_name_ls[@]}; do
  for j in ${!pgd_eps_str_list[@]}; do
    
    dilation_flag=""
    if [[ ${model_name_ls[$i]} = *'dc5'* ]]; then
      dilation_flag="--dilation"
    fi
    
    echo "Running AutoPGD attack on DETR-${model_name_ls[$i]} (KITTI) with eps=${pgd_eps_list[$j]}, iters=${pgd_iters}"
    

    python attack_Main_v4_modified.py \
      --backbone ${backbone_ls[$i]} \
      --resume ${model[$i]} \
      --num_classes $num_classes\
      $dilation_flag \
      --coco_path $path \
      --attack yes \
      --attack_type $attack \
      --pgd_eps ${pgd_eps_list[$j]} \
      --pgd_iters $pgd_iters \
      --autopgd_restarts $autopgd_restarts \
      --save_images True \
      --dataset_file $dataset_name \
      --save_images_path $output_dir
      > /home/anazeri/Transformer_ObjDet_Robustness/eval_results/${attack}/${dataset_name}/${attack}_${pgd_eps_str_list[$j]}_${dataset_name}_detr${model_name_ls[$i]}.txt 
      
  done
done












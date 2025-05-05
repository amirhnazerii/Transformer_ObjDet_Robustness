#### Main

'''

02/09/2024 added: parser.add_argument('--dataset', default=''
02/08/2024 New attack_Resnet50_coco.py (- CW attack func and param revised) downloaded from our Github and renamed to: attack_Main_v2.py



'''


import datetime
import json
import random
import time
from pathlib import Path
import torch.optim as optim
import utils2
from datasets.funcs import get_imgs_filenames, UnNormalize

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.funcs import get_imgs_filenames, UnNormalize
from engine import evaluate, train_one_epoch
from models import build_model

from models.detr import SetCriterion
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as T
import argparse

import math
from PIL import Image
import requests

import utils2    # Amir version modifed save_image func
import torch.optim as optim
import copy
import os

import numpy

#------------------------------------------------------------------------


# def get_args_parser():

if __name__ == '__main__':
    # Construct the argument parser.
    parser = argparse.ArgumentParser()
#     parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--attack', default='', type=str, choices=('yes', 'no'))
    parser.add_argument('--epsilon', default=0, type=float)
    parser.add_argument('--save_images', default='False', type=str, choices=('False', 'True'))
    parser.add_argument('--save_images_tensor', default='False', type=str, choices=('False', 'True'))
    parser.add_argument('--save_images_path', default='imgs/', type=str)
    parser.add_argument('--save_images_tensor_path', default='imgs/', type=str)
    parser.add_argument('--attack_type', default='', type=str)
    parser.add_argument('--pgd_eps', default=10/255, type=float)
    parser.add_argument('--pgd_iters', default=100, type=int)
    parser.add_argument('--cw_c', default=3, type=float)
    parser.add_argument('--cw_kappa', default=0, type=float)
    parser.add_argument('--cw_iters', default=200, type=int)
    parser.add_argument('--cw_lr', default=0.01, type=float)
    parser.add_argument('--dataset', default='', type=str, 
                       help='either coco or kitti')
    
    
            
        
#     return parser


#------------------------------------------------------------------------




def fgsm_attack(img_tensors, epsilon, model, annotation, criterion, UnNorm):
        # Collect the element-wise sign of the data gradient
        img_tensors.tensors.requires_grad = True
        outputs= model(img_tensors)

        loss_dict = criterion(outputs, [annotation[0]])
        loss_ce= -loss_dict['loss_ce']
        # weight_dict = criterion.weight_dict
        #print('loss_ce: .............', loss_ce)
        # Calculate gradients of model in backward pass
        loss_ce.backward()
        img_grad = img_tensors.tensors.grad
        
        # Restore the data to its original scale:
        img_denorm = UnNorm(img_tensors.tensors.detach().cpu())
                        
        img_denorm = img_denorm.to(device)
    
        sign_data_grad = img_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
#         if args.save_images == 'True':
        perturbed_image = img_denorm + epsilon*sign_data_grad
#         else:
#             perturbed_image = img_tensors.tensors + epsilon*sign_data_grad
        # Adding clipping to maintain [0, 1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image


def pgd_attack(img_tensors, alpha, model, annotation, criterion, UnNorm, eps, iters):

        """
        img_denorm = UnNorm(img_tensors.tensors.detach().cpu())        
        img_denorm = img_denorm.to(device)
        final_noise = 0
        for i in range(iters):
            #print(img_tensors.tensors)

            # Restore the data to its original scale:
            
            outputs= model(img_tensors)
            model.zero_grad()
            loss_dict = criterion(outputs, [annotation[0]])
            loss_ce= -loss_dict['loss_ce']
            # weight_dict = criterion.weight_dict
            #print('loss_ce: .............', loss_ce)
            # Calculate gradients of model in backward pass
            loss_ce.backward()
            img_grad = img_tensors.tensors.grad
        
            sign_data_grad = img_grad.sign()

            noise = torch.clamp(alpha*sign_data_grad, min=-eps, max=eps)
            perturbed_image = torch.clamp(img_tensors.tensors + noise, min=0, max=1).detach_()
            img_tensors.tensors = perturbed_image
            img_tensors.tensors.requires_grad = True
            final_noise += noise
        """
        img_denorm = UnNorm(img_tensors.tensors.detach().cpu())        
        img_denorm = img_denorm.to(device)
        final_noise = 0
        img_tensors.tensors.requires_grad = True
        for i in range(iters):
            #print(img_tensors.tensors)

            # Restore the data to its original scale:
            
            outputs= model(img_tensors)
            model.zero_grad()
            loss_dict = criterion(outputs, [annotation[0]])
            loss_ce= -loss_dict['loss_ce']
            # weight_dict = criterion.weight_dict
            #print('loss_ce: .............', loss_ce)
            # Calculate gradients of model in backward pass
            loss_ce.backward()
            img_grad = img_tensors.tensors.grad
        
            sign_data_grad = img_grad.sign()

            noise = torch.clamp(alpha*sign_data_grad, min=-eps, max=eps)
            perturbed_image = torch.clamp(img_tensors.tensors + noise, min=0, max=1).detach_()
            img_tensors.tensors = perturbed_image
            img_tensors.tensors.requires_grad = True
            final_noise += noise
        
#         if args.save_images == 'True':
        saved_image = torch.clamp(img_denorm + final_noise, min=0, max=1)
#         else:
#             saved_image = torch.clamp(img_tensors.tensors + final_noise, min=0, max=1)
        return saved_image


def cw_attack(img_tensors, learning_rate, model, annotation, criterion, UnNorm, c, kappa, iters):        

        if args.save_images == 'True':
            img_denorm = UnNorm(img_tensors.tensors.detach().cpu())        
            img_denorm = img_denorm.to(device)
            img_tensors.tensors = img_denorm
        img_tensors.tensors.requires_grad = True
        # Define f-function
        def f(x) :
            outputs = model(x)
            num_classes = 91 #number of classes + 1
            src_logits = outputs['pred_logits']
            outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
            indices = criterion.matcher(outputs_without_aux, [annotation[0]])
            idx = criterion._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip([annotation[0]], indices)])
            target_classes_o = target_classes_o.to(device)
            target_classes = torch.full(src_logits.shape[:2], num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
            one_hot_labels = F.one_hot(target_classes.to(src_logits.device))
    
            i, _ = torch.max((1-one_hot_labels)*src_logits, dim=2)
            j = torch.masked_select(src_logits, one_hot_labels.bool())
            
            return torch.clamp(j-i, min=-kappa)

        w = torch.zeros_like(img_tensors.tensors, requires_grad=True).to(device)
        optimizer = optim.Adam([w], lr=learning_rate)
        prev = 1e10

        for step in range(iters) :
            a = 1/2*(nn.Tanh()(w) + 1)
            adv_tensors = copy.deepcopy(img_tensors)
            adv_tensors.tensors = a
            
            loss1 = nn.MSELoss(reduction='sum')(a, img_tensors.tensors)
            loss2 = torch.sum(c*f(adv_tensors))
    
            cost = loss1 + loss2
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
    
            # Early Stop when loss does not converge.
            if step % (iters//10) == 0 :
                if cost > prev :
                    print('Attack Stopped due to CONVERGENCE....')
                    return a
                    
                prev = cost
            
            print('- Learning Progress : %2.2f %%        ' %((step+1)/iters*100), end='\r')

        perturbed_img = 1/2*(nn.Tanh()(w) + 1)
        return perturbed_img



class Adv_Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, images, annotations):
        'Initialization'
        self.images = images
        self.annotations = annotations
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.annotations)
    
    def __getitem__(self, index):
    
        # Load data and get label
        img = self.images[index]
        label = self.annotations[index]
    
        return img, label
    
    
#------------------------------------------------------------------------
# args = vars(parser.parse_args())


# parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
# args = parser.parse_args(args=[])
args = parser.parse_args()

args.batch_size = 1
args.no_aux_loss = True
args.eval = True
# args.backbone = 'resnet101'
# args.backbone = 'resnet50'
# args.dilation = True
# args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
# args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-dc5-f0fb7ef5.pth'
# args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r101-2c7b67e5.pth'

# args.adv_img_path = '/scratch1/anazeri/val2017_coco_noresize_detr_r101_adv02/'
args.adv_img_path= None

#------------------------------------------------------------------------


# #-----------------------------------
if args.save_images == "True":
    # save image for transferability assessment.
    from datasets import build_dataset2   # build_dataset2 imports imgs with their orig size without resizing in prepross part.
    dataset_val = build_dataset2(image_set='val', args=args)
else:
    dataset_val = build_dataset(image_set='val', args=args)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
base_ds = get_coco_api_from_dataset(dataset_val)

device = torch.device(args.device)
seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
model, criterion, postprocessors = build_model(args)
model.to(device)
model_without_ddp = model
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

param_dicts = [
    {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": args.lr_backbone,
    },
]
optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                              weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

output_dir = Path(args.output_dir)
if args.resume:
    if args.resume.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(args.resume, map_location='cpu')
        
    model_without_ddp.load_state_dict(checkpoint['model'])
    
    if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

for param in model.parameters():
    param.requires_grad = False


    

    
# from functools import partial
# def attn_drop_mask_grad(module, grad_innn, grad_out, gamma):
# #     print(module)
# #     print(grad_innn)
# #     # print(grad_out)
# #     print(type(grad_innn))      # tuple
# #     print(grad_innn[0].size())  # torch.Size([197, 192])

#     mask = torch.ones_like(grad_innn[0]) * gamma
# #     print(f'mask size ....: {mask.size()}, mask....: {mask}' )   #torch.Size([197, 192])
#     masked= mask * grad_innn[0][:]
# #     print(f'mask * grad_innn[0][:] size ....: {masked.size()}, masked....: {masked}' )     #torch.Size([197, 192])
#     return (masked, )

# drop_hook_func = partial(attn_drop_mask_grad, gamma= 1000)
# model.transformer.encoder.register_full_backward_hook(drop_hook_func)

# drop_hook_func2 = partial(attn_drop_mask_grad, gamma= 1000)
# model.backbone.register_full_backward_hook(drop_hook_func2)

#------------------------------------------------------------------------
#### get original images sizes
get_imgs_sizes = False
if get_imgs_sizes:
    from datasets.funcs import get_imgs_hw
    get_imgs_hw(Image_set= 'val', Args= args, save=True)
#------------------------------------------------------------------------




if args.attack== 'yes':
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')

        import torchvision.transforms.functional as FF
        import utils2
        from datasets.funcs import get_imgs_filenames, UnNormalize
        
    
        imgs_filenames_list= get_imgs_filenames(Image_set= 'val', Args= args )
        UnNorm= UnNormalize(mean= [0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
        

        if args.dataset == 'coco':
            imgs_hw_list = np.loadtxt("raw_imgs_hw_np.csv", delimiter=",", dtype=int)
        elif args.dataset == 'kitti':
            imgs_hw_list = np.loadtxt("KITTI_raw_imgs_hw.csv", delimiter=",", dtype=int)

        
        img_tensors_list = []
        annotation_list = []
        coco_anno_list = []
        img_grads_list = []  
        save_grads = False
        
            
        if args.save_images == "True":
                os.mkdir(args.save_images_path) # make a directory to save adv images
                   
        if args.save_images_tensor == "True":
                os.mkdir(args.save_images_tensor_path) # make a directory to save adv images tensors
        

        for i, (img, annotation) in enumerate(data_loader_val):
            
                
                
                img_tensors = img.to(device)
                img_tensors.tensors.requires_grad = True

                
                """
                outputs= model(img_tensors)
                
                loss_dict = criterion(outputs, [annotation[0]])
                loss_ce= -loss_dict['loss_ce']
                # weight_dict = criterion.weight_dict
                #print('loss_ce: .............', loss_ce)
                # Calculate gradients of model in backward pass
                loss_ce.backward()
                img_grad = img_tensors.tensors.grad
                
                # Restore the data to its original scale:
                img_denorm = UnNorm(img_tensors.tensors[0].detach().cpu())
                                
                img_denorm = img_denorm.to(device)
                """
            
                # Call FGSM Attack:
                #perturbed_img = fgsm_attack(img_denorm, args.epsilon, img_grad) # size: [1, 3, 800, 1201]
                if args.attack_type == 'fgsm':
                    perturbed_img = fgsm_attack(img_tensors, args.epsilon, model, annotation, criterion, UnNorm) # size: [1, 3, 800, 1201]

                if args.attack_type == 'pgd':
                    perturbed_img = pgd_attack(img_tensors, args.epsilon, model, annotation, criterion, UnNorm, args.pgd_eps, args.pgd_iters) # size: [1, 3, 800, 1201]

                if args.attack_type == 'cw':
                    perturbed_img = cw_attack(img_tensors, args.cw_lr, model, annotation, criterion, UnNorm, args.cw_c, args.cw_kappa, args.cw_iters) # size: [1, 3, 800, 1201]


            
                perturbed_img_norm = FF.normalize(perturbed_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                img_tensors.tensors = perturbed_img_norm  
                
                if args.save_images =="True":
                    perturbed_img_resiz = FF.resize(perturbed_img, imgs_hw_list[i] )
                    utils2.save_image(perturbed_img_resiz, args.save_images_path + imgs_filenames_list[i]+ ".jpg")
#                     "/scratch1/anazeri/val2017_coco_origsize_detr_r50DC5_adv02/"
                    del perturbed_img_resiz
                
                if save_grads:
                    img_grads_list.append(FF.resize(img_grad, imgs_hw_list[i]).detach().cpu())
                    # print(img_grads_list[0])  
                    # print(img_grads_list[0].size())
                    
                if args.save_images_tensor =="True":
                        with open(args.save_images_tensor_path + imgs_filenames_list[i]+ ".npz", 'wb') as f:
                            np.save(f, perturbed_img_norm.detach().cpu())
                
                
                img_tensors_list.append(img_tensors.tensors[0].detach().cpu())
                annotation_list.append(annotation[0])
                if i % 100 == 0:
                      print("%d Finished" % i)
                if i == 50:
                    break
                    
                del img
                del annotation 
                del img_tensors
                del perturbed_img
                del perturbed_img_norm
        if save_grads:
            imgs_grads_dict= dict(list(enumerate(img_grads_list)))
            torch.save(imgs_grads_dict, "/scratch1/anazeri/imgs_grads_tot")
        

        adv_dataset_val = Adv_Dataset(img_tensors_list, annotation_list)
        adv_sampler_val = torch.utils.data.SequentialSampler(adv_dataset_val)
        adv_data_loader_val = DataLoader(adv_dataset_val, args.batch_size, sampler=adv_sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

     
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                      adv_data_loader_val, base_ds, device, args.output_dir)

    

#------------------------------------------------------------------------


#------------------------------------------------------------------------



# test_on_clean_data = True

if args.attack == 'no':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    import torchvision.transforms.functional as F
    from datasets.funcs import get_imgs_filenames, UnNormalize
    
    
    from datasets.funcs import get_imgs_filenames, UnNormalize
    imgs_filenames_list= get_imgs_filenames(Image_set= 'val', Args= args )


    
#    def iterator(data_loader_val):
#        img_tensors_list =[]
#        annotation_list = []
#        for i, (img, annotation) in enumerate(data_loader_val):
#        
#            
#            img_np = np.load(args.save_images_tensor_path + imgs_filenames_list[i]+ ".npz")
#            
#            img.tensors = torch.from_numpy(img_np)
#    
#            img_tensors_list.append(img.tensors[0]) #.detach().cpu())    
#            annotation_list.append(annotation[0])
#            if i % 100 == 0:
#                  print("%d Finished" % i)
#            if i == 4700:
#                break
#            del img_np
#            del img
#        return img_tensors_list, annotation_list
#
##    adv_dataset_val = Adv_Dataset(img_tensors_list, annotation_list)
#    adv_dataset_val = Adv_Dataset(iterator(data_loader_val)[0], iterator(data_loader_val)[1])
#
#    adv_sampler_val = torch.utils.data.SequentialSampler(adv_dataset_val)
#    adv_data_loader_val = DataLoader(adv_dataset_val, args.batch_size, sampler=adv_sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
#
#     
#    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
#                                                      adv_data_loader_val, base_ds, device, args.output_dir)

    
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                         data_loader_val, base_ds, device, args.output_dir)

del img_tensors_list   
del annotation_list   
del adv_dataset_val    
del adv_data_loader_val   
    
del model

torch.cuda.empty_cache()




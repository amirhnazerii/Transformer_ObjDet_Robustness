import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
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


#------------------------------------------------------------------------


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
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
    
    
    parser.add_argument('--attack', default='', type=str)
    
    
    ### Transferability parser:
    
    parser.add_argument(
    '--data', 
    default='data_configs/test_image_config.yaml',
    help='(optional) path to the data config file'
    )
    parser.add_argument(
        '-m', '--model_transfer', 
        default='fasterrcnn_resnet50_fpn',
        help='name of the model to be tested under transfered attack'
    )
    parser.add_argument(
        '-mw', '--weights', 
        default=None,
        help='path to trained checkpoint weights if providing custom YAML file'
    )
    parser.add_argument(
        '-ims', '--imgsz', 
        default=640, 
        type=int, 
        help='image size to feed to the network'
    )
    # parser.add_argument(
    #     '-w', '--workers', default=4, type=int,
    #     help='number of workers for data processing/transforms/augmentations'
    # )
    # parser.add_argument(
    #     '-b', '--batch', 
    #     default=8, 
    #     type=int, 
    #     help='batch size to load the data'
    # )
    # parser.add_argument(
    #     '-d', '--device', 
    #     default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    #     help='computation/training device, default is GPU if GPU present'
    # )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='show class-wise mAP'
    )
    parser.add_argument(
        '-st', '--square-training',
        dest='square_training',
        action='store_true',
        help='Resize images to square shape instead of aspect ratio resizing \
              for single image training. For mosaic training, this resizes \
              single images to square shape first then puts them on a \
              square canvas.'
    )
    
    
    
    
    
    return parser


#------------------------------------------------------------------------


# COCO classes
CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size, device):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(device)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        print(CLASSES[cl])
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def fgsm_attack(image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image
    
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


parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args(args=[])
args.batch_size = 1
args.no_aux_loss = True
args.eval = True
# args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
args.backbone = 'resnet50'
args.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
args.coco_path = '/home/anazeri/Adv_ViT_OD/datasets/coco_dataset2017/'


#------------------------------------------------------------------------



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


    
#------------------------------------------------------------------------

# attack = True

args.attack='no'

if args.attack== 'yes':

        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        epsilon = 0.1
        img_tensors_list = []
        annotation_list = []
        for i, (img, annotation) in enumerate(data_loader_val):
            
            
                img_tensors = img.to(device)
                img_tensors.tensors.requires_grad = True
                outputs= model(img_tensors)
                """
                im_size  = [img_tensors.tensors.size(dim = 3),img_tensors.tensors.size(dim = 2)]
                # keep only predictions with 0.7+ confidence
                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.7
                # convert boxes from [0; 1] to image scales
                bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im_size, device)
                plot_results(img_tensors.tensors[0].detach().cpu().permute(1, 2, 0), probas[keep], bboxes_scaled)
                """
                loss_dict = criterion(outputs, [annotation[0]])
                loss_ce= -loss_dict['loss_ce']
                # weight_dict = criterion.weight_dict
                #print('loss_ce: .............', loss_ce)
                # Calculate gradients of model in backward pass
                loss_ce.backward()
                img_grad = img_tensors.tensors.grad
                #print(img_tensors.tensors.is_leaf)
                #print("img_gradddddddddddddddddddd:  ", img_grad)
                # Call FGSM Attack
                perturbed_img = fgsm_attack(img_tensors.tensors, epsilon, img_grad)
                #adv_tensors = copy.copy(img_tensors)
                img_tensors.tensors = perturbed_img

                """
                adv_outputs = model(adv_tensors)
                # keep only predictions with 0.7+ confidence
                adv_probas = adv_outputs['pred_logits'].softmax(-1)[0, :, :-1]
                adv_keep = adv_probas.max(-1).values > 0.7
                # convert boxes from [0; 1] to image scales
                adv_bboxes_scaled = rescale_bboxes(adv_outputs['pred_boxes'][0, adv_keep], im_size, device)
                plot_results(perturbed_img[0].detach().cpu().permute(1, 2, 0), adv_probas[keep], adv_bboxes_scaled)
                """
                # print("type of img_tensors.tensors[0].detach().cpu()...............................................", type(img_tensors.tensors[0])) == <class 'torch.Tensor'>
                img_tensors_list.append(img_tensors.tensors[0].detach().cpu())
                annotation_list.append(annotation[0])
                print(i)
                # if i % 100 == 0:
                #     print("%d Finished" % i)
                
                if i == 10:
                    break
            
                del img
                del annotation

        adv_dataset_val = Adv_Dataset(img_tensors_list, annotation_list)
        print("type of adv_dataset_val...............................................", type(adv_dataset_val))
        print("inside of adv_dataset_val[0]...............................................", adv_dataset_val[0])

        adv_sampler_val = torch.utils.data.SequentialSampler(adv_dataset_val)
        adv_data_loader_val = DataLoader(adv_dataset_val, args.batch_size, sampler=adv_sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        print("type of adv_data_loader_val...............................................", type(adv_data_loader_val))


        """
        for i, (adv_img, annotation) in enumerate(adv_data_loader_val):
            adv_img_tensors = adv_img.to(device)
            adv_outputs= model(adv_img_tensors)
            im_size  = [adv_img_tensors.tensors.size(dim = 3),adv_img_tensors.tensors.size(dim = 2)]
            # keep only predictions with 0.7+ confidence
            probas = adv_outputs['pred_logits'].softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values > 0.7
            # convert boxes from [0; 1] to image scales
            adv_bboxes_scaled = rescale_bboxes(adv_outputs['pred_boxes'][0, keep], im_size, device)
            plot_results(adv_img_tensors.tensors[0].detach().cpu().permute(1, 2, 0), probas[keep], adv_bboxes_scaled)
        """
        # test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
        #                                               adv_data_loader_val, base_ds, device, args.output_dir)

    

#------------------------------------------------------------------------


#------------------------------------------------------------------------



# test_on_clean_data = True

if args.attack == 'no':

    dataset_cut = True
    if dataset_cut:
    
        ## test on Original clean dataset
        def dataset_cut(data_loader_val):
            img_clean_list =[]
            annotation_clean_list = []
            for i, (img, annotation) in enumerate(data_loader_val):

                print(i)
                img= img.to(device)
                if i <= 30:
                # if i > 6480:

                    img_clean_list.append(img.tensors[0].detach().cpu())
                    annotation_clean_list.append(annotation[0])
                    
                    if i==30:
                        break
                        
                del img
                del annotation

            return img_clean_list, annotation_clean_list

        img_clean_list, annotation_clean_list = dataset_cut(data_loader_val)

            # else:
            #     break


        print('img_clean_list size:.................', len(img_clean_list) )
        _dataset_val = Adv_Dataset(img_clean_list, annotation_clean_list)
        _sampler_val = torch.utils.data.SequentialSampler(_dataset_val)
        
        #-----------------------Edit-----------------#
        ## To avoid: TypeError: 'NestedTensor' object is not iterable. USE: collate_fn_FasterRCNN
        _data_loader_val = DataLoader(_dataset_val, args.batch_size, sampler=_sampler_val, drop_last=False, collate_fn=utils.collate_fn_FasterRCNN, num_workers=args.num_workers)
        #--------------------------------------------#
        # test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
        #                                           _data_loader_val, base_ds, device, args.output_dir)
    

    # test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                               data_loader_val, base_ds, device, args.output_dir)

    
    
Transfer_attack = False
if Transfer_attack:
    
        from torch_utils.engine import evaluate as evaluate_original_torch
        from datasets_py import (
    create_valid_dataset, create_valid_loader
)
        import yaml
        args.data = "/home/anazeri/fasterrcnn-pytorch-training-pipeline/data_configs/coco2017.yaml"
        args.model_transfer = "fasterrcnn_resnet50_fpn_v2"
        args.verbose = True 
    
        #### Transferability Detr_R50 to FasterRCNN_R50FPN:

        from models_FasterRCNN.create_fasterrcnn_model import create_model
        
        # Load the data configurations
        with open(args.data) as file:
            data_configs = yaml.safe_load(file)
                       
        # Validation settings and constants.
        try: # Use test images if present.
            VALID_DIR_IMAGES = data_configs['TEST_DIR_IMAGES']
            VALID_DIR_LABELS = data_configs['TEST_DIR_LABELS']
        except: # Else use the validation images.
            VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
            VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
        #----------------- edited ------------------#
        NUM_WORKERS = args.num_workers   
        #----------------- edited ------------------#
        DEVICE = args.device
        BATCH_SIZE = args.batch_size

        # Model configurations
        IMAGE_SIZE = args.imgsz
        
        # valid_loader= create_valid_loader(_dataset_val, BATCH_SIZE, NUM_WORKERS)

        # Load the pretrained model
        create_model = create_model[args.model_transfer]
        if args.weights is None:
            try:
                model_2, coco_model = create_model(num_classes=NUM_CLASSES, coco_model=True)
            except:
                model_2 = create_model(num_classes=NUM_CLASSES, coco_model=True)
            if coco_model:
                COCO_91_CLASSES = data_configs['COCO_91_CLASSES']
                valid_dataset = create_valid_dataset(
                    VALID_DIR_IMAGES, 
                    VALID_DIR_LABELS, 
                    IMAGE_SIZE, 
                    COCO_91_CLASSES, 
                    square_training=args.square_training
                )
        model_2.to(DEVICE).eval()
        
        
        # print("valid_dataset[0]..................", valid_dataset[0])
        # print("valid_dataset[1]..................", valid_dataset[1])
        
#         for i, img_annotation in enumerate(valid_dataset):
            
#             # print("img.................", valid_dataset.img)
        
#             # print(valid_dataset[i])
#             # print(valid_dataset[i][0])
#             valid_dataset[i][0] = img_clean_list[i]

            
        
#             if i==30:
#                 break
        
        valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)
        
        
        stats, val_pred_image = evaluate_original_torch(
                                    model_2, 
                                    valid_loader, 
                                    device=DEVICE,
                                    classes=CLASSES)
        print(val_pred_image)
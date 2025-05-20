###

'''
datasets/funcs.py does not exist in original detr https://github.com/facebookresearch/detr/tree/main/datasets

Created to help save attacked images correctly.
e.g. get_imgs_hw()

'''


###


from pathlib import Path

import torch
import torch.utils.data
import torchvision




class CocoDetection2(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file):
        super(CocoDetection2, self).__init__(img_folder, ann_file)
        
        # from pycocotools.coco import COCO
        # self.coco = COCO(annFile)
        self.imgs_filenames_raw= self.ids    # get id


def build2(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection2(img_folder, ann_file)
    imgs_filenames_raw =dataset.imgs_filenames_raw
    # coco         =dataset.coco.loadImgs(id)[0]["file_name"]
    return imgs_filenames_raw   # imgs_filenames_raw : list of raw filenames




def get_imgs_filenames(Image_set, Args):
    from .funcs import build2
    imgs_filenames_raw= build2(image_set=Image_set, args=Args) # imgs_filenames_raw : list

    imgs_filenames_list = []
    
    if Args.dataset == 'coco':
        for i in range(len(imgs_filenames_raw)):
            img_filename_raw = str(imgs_filenames_raw[i])
            zeros_len = 12 - len(img_filename_raw)
            zeros = zeros_len*"0"
            img_filename = zeros+img_filename_raw
            imgs_filenames_list.append(img_filename)
    elif Args.dataset == 'kitti':
        for i in range(len(imgs_filenames_raw)):
            img_filename_raw = str(imgs_filenames_raw[i]-1)  # in kitti: filename integer = id -1
            zeros_len = 6 - len(img_filename_raw)
            zeros = zeros_len*"0"
            img_filename = zeros+img_filename_raw
            imgs_filenames_list.append(img_filename)
        
    return imgs_filenames_list


def get_imgs_hw(Image_set, Args, raw_imgs_hw_name= None, save= False):
    
    """
    
    return:
    ------------
    raw_imgs_hw_list: np, size: [N,2] N: num of imgs. 
    """
    import numpy as np
    from datasets import build_dataset2
    dataset_val_raw = build_dataset2(image_set=Image_set, args=Args)    # resize is off in build_dataset2
    raw_imgs_hw_list = []
    for i, (img, annotation) in enumerate(dataset_val_raw):

            # print(img.size())
            raw_imgs_hw_list.append(img.size()[-2:])  # img: [c, w, h]
            # plt.imshow(img.permute(1, 2, 0))
            # plt.show()
            
            # if i == 10:
            #     break                
    raw_imgs_hw_list= np.array(raw_imgs_hw_list).astype(int)
    if save:
        np.savetxt(raw_imgs_hw_name, raw_imgs_hw_list, 
              delimiter = ",")
    return raw_imgs_hw_list
    
    
    
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
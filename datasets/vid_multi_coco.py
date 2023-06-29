from PIL import Image
import sys
import numpy as np
import random

from .vid_coco import VIDDataset, make_vid_transforms
# from mega_core.config import cfg
import datasets.transforms as T

class VIDMULTIDataset(VIDDataset):
    def __init__(self, image_set, img_dir, json_file, transforms, is_train=True, cfg=None):
        super().__init__(image_set, img_dir, json_file, transforms, is_train=is_train, cfg=cfg)
        self.cfg = cfg
        self.max_offset = 12
        self.min_offset = -12
        self.ref_num_local = 2

        self.test_with_one_img = False
        self.test_ref_nums = 2
        self.test_max_offset = 12
        self.test_min_offset = -12

        if cfg is not None:

            self.test_with_one_img = cfg.TEST.test_with_one_img
            self.test_ref_nums = cfg.TEST.test_ref_nums
            self.test_max_offset = cfg.TEST.test_max_offset
            self.test_min_offset = cfg.TEST.test_min_offset


        if cfg is not None:
            self.max_offset = cfg.DATASET.max_offset
            self.min_offset = cfg.DATASET.min_offset
            self.ref_num_local = cfg.DATASET.ref_num_local

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs_l = []
        if hasattr(self, "pattern"):
            offsets = np.random.choice(self.max_offset - self.min_offset + 1,
                                       self.ref_num_local, replace=False) + self.min_offset
            for i in range(len(offsets)):
                ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[self.frame_id[idx]][ref_id]
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_l.append(img_ref)

        else:
            for i in range(self.ref_num_local):
                img_refs_l.append(img.copy())

        target = self.get_groundtruth(idx)
        p_dict = None

        if self.transforms is not None:
            img, target = self.transforms(img, target, p_dict)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None, p_dict)
        images = {}
        images["cur"] = img  # to make a list
        images["ref_l"] = img_refs_l

        return images, target

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = self.frame_seg_id[idx]

        if self.test_with_one_img:

            img_refs_l = []
            # reading other images of the queue (not necessary to be the last one, but last one here)
            ref_id = min(max(self.frame_seg_len[idx] - 1, 0), frame_id + self.max_offset)
            ref_filename = self.pattern[self.frame_id[idx]][ref_id]
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)

        else:
            img_refs_l = self.get_ref_imgs(idx)

        img_refs_g = []

        target = self.get_groundtruth(idx)
        # target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l

        return images, target

    def get_ref_imgs(self, idx):
        filename = self.image_set_index[idx]
        frame_id = self.frame_seg_id[idx]
        ref_id_list = []
        ref_start_id = frame_id + self.test_min_offset
        ref_end_id = frame_id + self.test_max_offset

        interval = (ref_end_id - ref_start_id) // (self.test_ref_nums - 1)

        for i in range(ref_start_id, ref_end_id + 1, interval):
            # print(i)
            ref_id_list.append(min(max(0, i), self.frame_seg_len[idx] - 1))

        img_refs_l = []

        for ref_id in ref_id_list:
            # print(ref_id)
            ref_filename = self.pattern[self.frame_id[idx]][ref_id]
            img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
            img_refs_l.append(img_ref)

        return img_refs_l



def build_vitmulti_transforms(is_train):
    # todo fixme add data augmantation
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])



def build_vidmulti(image_set, cfg, transforms=build_vitmulti_transforms(True)):

    is_train = (image_set == 'train')
    if is_train:
        dataset = VIDMULTIDataset(
        image_set = image_set,
        img_dir = ,
        anno_path = "train.json",
        transforms = transforms, # make_vid_transforms('train'),
        is_train = is_train,
        cfg=cfg
        )
    else:
        dataset = VIDMULTIDataset(
        image_set = image_set,
        img_dir = "/content/drive/MyDrive/YOLOv5 (1)/datasets/Worms_NewAlignment_Parafilm/validation/images/",
        anno_path = "val.json",
        transforms = transforms, # make_vid_transforms('val'),
        is_train = is_train,
        cfg=cfg
        )

    return dataset

'''
def build_detmulti(image_set, cfg):

    is_train = (image_set == 'train')
    assert is_train is True  # no validation dataset
    dataset = VIDMULTIDataset(
    image_set = "DET_train_30classes",
    img_dir = "/dataset/public/ilsvrc2015/Data/DET",
    anno_path = "/dataset/public/ilsvrc2015/Annotations/DET",
    img_index = "/data1/wanghan20/Prj/VODETR/datasets/split_file/DET_train_30classes.txt",
    transforms=build_vitmulti_transforms(True),
    is_train=is_train,
    cfg=cfg
    )
    return dataset
'''
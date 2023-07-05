import os
import pickle

import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np
from util.misc import is_main_process
import xml.etree.cElementTree as ET
import datasets.transforms as T
import json

import pycocotools.coco as coco

class VIDDataset(torch.utils.data.Dataset):
    '''
    classes = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']
    classes_map = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']
                    '''
    classes = {0:'__background__', 15:'dead_worms', 16:'worms'}

    def __init__(self, image_set, img_dir, json_file, transforms, is_train = True, coco_anno_path = None, cfg = None):

        '''
        old:
        dataset = VIDDataset(
        image_set = "VID_train_15frames",
        data_dir = "/dataset/public",
        img_dir = "/dataset/public/ilsvrc2015/Data/VID",
        anno_path = "/dataset/public/ilsvrc2015/Annotations/VID",
        img_index = "/dataset/public/ilsvrc2015/ImageSets/VID_train_15frames.txt",
        transforms=make_vid_transforms('train'),
        is_train=is_train
        )
        
        new:
        dataset = COCODataset(
        image_set
        img_dir
        json_file
        transforms
        )
        '''

        #self.det_vid = image_set.split("_")[0]
        self.image_set = image_set # = train/val/test
        self.transforms = transforms
        self.cfg = cfg

        self.img_dir = img_dir # = path
        #self.anno_path = anno_path
        #self.img_index = img_index 
        self.json_file = json_file 

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s.jpg")
        #self._anno_path = os.path.join(self.anno_path, "%s.xml")

        with open(self.img_dir + "/" + self.json_file) as f:
            self.coco_data = json.load(f)

        self.image_set_index = [image["file_name"].replace(".jpg", "") for image in self.coco_data["images"]]
        self.frame_id = [image["video_id"] for image in self.coco_data["images"]]
        self.frame_seg_id = [image["frame_id"] for image in self.coco_data["images"]]
        self.frame_seg_len = [self.coco_data["videos"][image["video_id"] - 1]["nframes"] for image in self.coco_data["images"]]

        # pattern[video_id][frame_id][name] = "file_name"
        self.pattern = {}
        for image in self.coco_data["images"]:
            if image["video_id"] not in self.pattern:
                self.pattern[image["video_id"]] = {}
            self.pattern[image["video_id"]][image["frame_id"]] = image["file_name"].replace(".jpg", "")


        #self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        #self.categories = dict(zip(range(len(self.classes)), self.classes))

        self.annos = self.per_image_annotation()

        coco_anno_path = self.img_dir + "/" + self.json_file
        self.coco = coco.COCO(coco_anno_path)

    '''
    def convert_to_coco(self):
        # print(len(self.annos), len(self.image_set_index))
        assert len(self.annos) == len(self.image_set_index)
        print('converting to coco style ...')

        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'person'}]}

        ann_cnt = 0
        for idx in range(len(self.image_set_index)):
            filename = self.image_set_index[idx]
            image_cnt = idx + 1
            image_info = {'file_name': filename,
                          'id': image_cnt}
            out['images'].append(image_info)
            target = self.get_groundtruth(idx)

            for i in range(len(target["boxes"])):
                bbox_xyxy = list(np.array(target["boxes"][i], dtype=np.float32).copy())
                label = int(target["labels"][i])
                area = float(target["areas"][i])
                bbox_xywh = list(np.array([bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2] - bbox_xyxy[0] + 1,
                                           bbox_xyxy[3] - bbox_xyxy[1] + 1]))

                # id is global ann id, bbox is xywh (exactly ltwh) style
                ann_cnt += 1
                ann = {'id': ann_cnt,
                       'category_id': label,
                       'image_id': image_cnt,
                       'bbox': bbox_xywh,
                       'area': area,
                       'iscrowd': 0
                       }  # fixme iscrowd
                out['annotations'].append(ann)
        print('loaded {} images and {} samples'.format(
            len(out['images']), len(out['annotations'])))
            # target = {"boxes": boxes, "orig_size": size, "size": size, "labels": labels,
            #           "image_id": torch.tensor(idx)}  # todo image_id
        return out
    '''

    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        target = self.get_groundtruth(idx)
        # target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _get_test(self, idx):
        return self._get_train(idx)

    def __len__(self):
        return len(self.image_set_index)

    '''
    def filter_annotation(self):
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            if is_main_process():
                print("{}'s keep information loaded from {}".format(self.det_vid, cache_file))
            return keep

        keep = np.zeros((len(self)), dtype=np.bool)
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had filtered {} images".format(idx))

            filename = self.image_set_index[idx]

            tree = ET.parse(self._anno_path % filename).getroot()
            objs = tree.findall("object")
            keep[idx] = False if len(objs) == 0 else True
        print("Had filtered {} images".format(len(self)))

        if is_main_process():
            with open(cache_file, "wb") as fid:
                pickle.dump(keep, fid)
            print("Saving {}'s keep information into {}".format(self.det_vid, cache_file))

        return keep

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox =obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            if is_main_process():
                print("{}'s annotation information loaded from {}".format(self.det_vid, cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 10000 == 0:
                    print("Had processed {} images".format(idx))

                filename = self.image_set_index[idx]

                tree = ET.parse(self._anno_path % filename).getroot()
                anno = self._preprocess_annotation(tree)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            if is_main_process():
                with open(cache_file, "wb") as fid:
                    pickle.dump(annos, fid)
                print("Saving {}'s annotation information into {}".format(self.det_vid, cache_file))

        return annos
    '''

    def get_img_info(self, idx):
        im_info = self.coco_data["images"][idx]
        return {"height": im_info["height"], "width": im_info["width"]}

    def per_image_annotation(self):
        annotations = self.coco_data['annotations']
        images = self.coco_data['images']
        piannotations = []
        for image in images:
            piannotation = dict()
            piannotation["im_info"] = image["height"], image["width"]
            idx = image["id"]
            piannotation["boxes"] = []
            piannotation["labels"] = []
            piannotation["areas"] = []

            for annotation in annotations:
                if annotation["image_id"] == idx:
                    [Xmid, Ymid, Width, Height] = annotation["bbox"]
                    bbox = [Xmid - 1/2 * Width, Ymid - 1/2 * Height, Xmid + 1/2 * Width, Ymid + 1/2 * Height]
                    piannotation["boxes"].append(bbox)
                    piannotation["labels"].append(annotation["category_id"])
                    piannotation["areas"].append(annotation["area"])
            piannotations.append(piannotation)

        return piannotations

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        # cache_dir = os.path.join(self.data_dir, 'cache')
        cache_dir = self.cfg.DATASET.cache_dir[0]
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    def get_visualization(self, idx):
        filename = self.image_set_index[idx]

        img = cv2.imread(self._img_dir % filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        # target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        # file_name = self.image_set_index[idx]
        anno = self.annos[idx]

        height, width = anno["im_info"]
        boxes = anno["boxes"]
        labels = anno["labels"]

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # boxes[:, 2:] += boxes[:, :2]  # ??? bug
        boxes[:, 0::2].clamp_(min=0, max=width)
        boxes[:, 1::2].clamp_(min=0, max=height)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = torch.tensor(labels, dtype=torch.int64)
        labels = labels[keep]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        size = torch.tensor([height, width])  #  rechecked, is h w
        # target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        # target.add_field("labels", anno["labels"])
        target = {"boxes": boxes, "orig_size": size, "size": size, "labels": labels, "areas": areas,
                  "image_id": torch.tensor(idx + 1)}  # todo image_id

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return VIDDataset.classes[class_id]


def make_vid_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_vid(image_set, cfg):
    # use cfg file ?

    is_train = (image_set == 'train')
    if is_train:
        dataset = VIDDataset(
        image_set = "train",
        img_dir = cfg.DATASET.train_dir,
        anno_path = "train.json",
        transforms = make_vid_transforms('train'),
        is_train = is_train
        )
    else:
        dataset = VIDDataset(
        image_set = "val",
        img_dir = cfg.DATASET.val_dir,
        anno_path = "val.json",
        transforms = make_vid_transforms('val'),
        is_train = is_train
        )

    return dataset

'''
def build_det(image_set, cfg):

    is_train = (image_set == 'train')
    assert is_train is True  # no validation dataset
    dataset = VIDDataset(
    image_set = "DET_train_30classes",
    data_dir = "/dataset/public",
    img_dir = "/dataset/public/ilsvrc2015/Data/DET",
    anno_path = "/dataset/public/ilsvrc2015/Annotations/DET",
    img_index = "/data1/wanghan20/Prj/VODETR/datasets/split_file/DET_train_30classes.txt",
    transforms=make_vid_transforms('train'),
    is_train=is_train
    )
    return dataset
'''

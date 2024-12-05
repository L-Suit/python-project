# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET

import numpy as np
from mmengine.fileio import dump, list_from_file
from mmengine.utils import mkdir_or_exist, track_progress

from mmdet.evaluation import voc_classes

# 原始voc的类别标签
label_ids = {name: i for i, name in enumerate(voc_classes())}

# label_ids = { 0: 'rice leaf roller',
#               1: 'rice leaf caterpillar',
#               2: 'paddy stem maggot',
#               3: 'asiatic rice borer',
#               4: 'yellow rice borer',
#               5: 'rice gall midge',
#               6: 'Rice Stemfly',
#               7: 'brown plant hopper',
#               8: 'white backed plant hopper',
#               9: 'small brown plant hopper',
#               10: 'rice water weevil',
#               11: 'riceleafhopper',
#               12: 'grain spreader thrips',
#               13: 'rice shell pest',
#               14: 'grub',
#               15: 'mole cricket',
#               16: 'wireworm',
#               17: 'white margined moth',
#               18: 'black cutworm',
#               19: 'large cutworm',
#               20: 'yellow cutworm',
#               21: 'red spider',
#               22: 'corn borer',
#               23: 'army worm',
#               24: 'aphids',
#               25: 'Potosiabre vitarsis',
#               26: 'peach borer',
#               27: 'english grain aphid',
#               28: 'green bug',
#               29: 'bird cherry-oataphid',
#               30: 'wheat blossom midge',
#               31: 'penthaleus major',
#               32: 'longlegged spider mite',
#               33: 'wheat phloeothrips',
#               34: 'wheat sawfly',
#               35: 'cerodonta denticornis',
#               36: 'beet fly',
#               37: 'flea beetle',
#               38: 'cabbage army worm',
#               39: 'beet army worm',
#               40: 'Beet spot flies',
#               41: 'meadow moth',
#               42: 'beet weevil',
#               43: 'sericaorient alismots chulsky',
#               44: 'alfalfa weevil',
#               45: 'flax budworm',
#               46: 'alfalfa plant bug',
#               47: 'tarnished plant bug',
#               48: 'Locustoidea',
#               49: 'lytta polita',
#               50: 'legume blister beetle',
#               51: 'blister beetle',
#               52: 'therioaphis maculata Buckton',
#               53: 'odontothrips loti',
#               54: 'Thrips',
#               55: 'alfalfa seed chalcid',
#               56: 'Pieris canidia',
#               57: 'Apolygus lucorum',
#               58: 'Limacodidae',
#               59: 'Viteus vitifoliae',
#               60: 'Colomerus vitis',
#               61: 'Brevipoalpus lewisi McGregor',
#               62: 'oides decempunctata',
#               63: 'Polyphagotars onemus latus',
#               64: 'Pseudococcus comstocki Kuwana',
#               65: 'parathrene regalis',
#               66: 'Ampelophaga',
#               67: 'Lycorma delicatula',
#               68: 'Xylotrechus',
#               69: 'Cicadella viridis',
#               70: 'Miridae',
#               71: 'Trialeurodes vaporariorum',
#               72: 'Erythroneura apicalis',
#               73: 'Papilio xuthus',
#               74: 'Panonchus citri McGregor',
#               75: 'Phyllocoptes oleiverus ashmead',
#               76: 'Icerya purchasi Maskell',
#               77: 'Unaspis yanonensis',
#               78: 'Ceroplastes rubens',
#               79: 'Chrysomphalus aonidum',
#               80: 'Parlatoria zizyphus Lucus',
#               81: 'Nipaecoccus vastalor',
#               82: 'Aleurocanthus spiniferus',
#               83: 'Tetradacus c Bactrocera minax',
#               84: 'Dacus dorsalis(Hendel)',
#               85: 'Bactrocera tsuneonis',
#               86: 'Prodenia litura',
#               87: 'Adristyrannus',
#               88: 'Phyllocnistis citrella Stainton',
#               89: 'Toxoptera citricidus',
#               90: 'Toxoptera aurantii',
#               91: 'Aphis citricola Vander Goot',
#               92: 'Scirtothrips dorsalis Hood',
#               93: 'Dasineura sp',
#               94: 'Lawana imitata Melichar',
#               95: 'Salurnis marginella Guerr',
#               96: 'Deporaus marginatus Pascoe',
#               97: 'Chlumetia transversa',
#               98: 'Mango flat beak leafhopper',
#               99: 'Rhytidodera bowrinii white',
#               100: 'Sternochetus frigidus',
#               101: 'Cicadellidae'
# }

def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        # name = obj.find('name').text
        label = obj.find('name').text
        difficult = int(obj.find('difficult').text)
        bnd_box = obj.find('bndbox')
        bbox = [
            int(bnd_box.find('xmin').text),
            int(bnd_box.find('ymin').text),
            int(bnd_box.find('xmax').text),
            int(bnd_box.find('ymax').text)
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path,
                            f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, '
                  f'skip voc{year} {split}')
            return
        img_names = list_from_file(filelist)
        xml_paths = [
            osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml')
            for img_name in img_names
        ]
        img_paths = [
            f'VOC{year}/images/{img_name}.jpg' for img_name in img_names
        ]
        part_annotations = track_progress(parse_xml,
                                          list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='pkl',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])
    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')
    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'
    for year in years:
        if year == '2007':
            prefix = 'voc07'
        elif year == '2012':
            prefix = 'voc12'
        elif year == ['2007', '2012']:
            prefix = 'voc0712'
        # for split in ['train', 'val', 'trainval']:
        for split in ['trainval','test']:
            dataset_name = prefix + '_' + split
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, split,
                            osp.join(out_dir, dataset_name + out_fmt))
        if not isinstance(year, list):
            dataset_name = prefix + '_test'
            print(f'processing {dataset_name} ...')
            cvt_annotations(devkit_path, year, 'test',
                            osp.join(out_dir, dataset_name + out_fmt))
    print('Done!')


if __name__ == '__main__':
    main()

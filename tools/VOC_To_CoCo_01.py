# voc格式数据集转换为coco格式
import sys
import os
import json
import xml.etree.ElementTree as ET

START_BOUNDING_BOX_ID = 0

PRE_DEFINE_CATEGORIES = {
  'rice leaf roller': 1,
  'rice leaf caterpillar': 2,
  'paddy stem maggot': 3,
  'asiatic rice borer': 4,
  'yellow rice borer': 5,
  'rice gall midge': 6,
  'Rice Stemfly': 7,
  'brown plant hopper': 8,
  'white backed plant hopper': 9,
  'small brown plant hopper': 10,
  'rice water weevil': 11,
  'rice leafhopper': 12,
  'grain spreader thrips': 13,
  'rice shell pest': 14,
  'grub': 15,
  'mole cricket': 16,
  'wireworm': 17,
  'white margined moth': 18,
  'black cutworm': 19,
  'large cutworm': 20,
  'yellow cutworm': 21,
  'red spider': 22,
  'corn borer': 23,
  'army worm': 24,
  'aphids': 25,
  'Potosiabre vitarsis': 26,
  'peach borer': 27,
  'english grain aphid': 28,
  'green bug': 29,
  'bird cherry-oataphid': 30,
  'wheat blossom midge': 31,
  'penthaleus major': 32,
  'longlegged spider mite': 33,
  'wheat phloeothrips': 34,
  'wheat sawfly': 35,
  'cerodonta denticornis': 36,
  'beet fly': 37,
  'flea beetle': 38,
  'cabbage army worm': 39,
  'beet army worm': 40,
  'Beet spot flies': 41,
  'meadow moth': 42,
  'beet weevil': 43,
  'sericaorient alismots chulsky': 44,
  'alfalfa weevil': 45,
  'flax budworm': 46,
  'alfalfa plant bug': 47,
  'tarnished plant bug': 48,
  'Locustoidea': 49,
  'lytta polita': 50,
  'legume blister beetle': 51,
  'blister beetle': 52,
  'therioaphis maculata Buckton': 53,
  'odontothrips loti': 54,
  'Thrips': 55,
  'alfalfa seed chalcid': 56,
  'Pieris canidia': 57,
  'Apolygus lucorum': 58,
  'Limacodidae': 59,
  'Viteus vitifoliae': 60,
  'Colomerus vitis': 61,
  'Brevipoalpus lewisi McGregor': 62,
  'oides decempunctata': 63,
  'Polyphagotars onemus latus': 64,
  'Pseudococcus comstocki Kuwana': 65,
  'parathrene regalis': 66,
  'Ampelophaga': 67,
  'Lycorma delicatula': 68,
  'Xylotrechus': 69,
  'Cicadella viridis': 70,
  'Miridae': 71,
  'Trialeurodes vaporariorum': 72,
  'Erythroneura apicalis': 73,
  'Papilio xuthus': 74,
  'Panonchus citri McGregor': 75,
  'Phyllocoptes oleiverus ashmead': 76,
  'Icerya purchasi Maskell': 77,
  'Unaspis yanonensis': 78,
  'Ceroplastes rubens': 79,
  'Chrysomphalus aonidum': 80,
  'Parlatoria zizyphus Lucus': 81,
  'Nipaecoccus vastalor': 82,
  'Aleurocanthus spiniferus': 83,
  'Tetradacus c Bactrocera minax': 84,
  'Dacus dorsalis(Hendel)': 85,
  'Bactrocera tsuneonis': 86,
  'Prodenia litura': 87,
  'Adristyrannus': 88,
  'Phyllocnistis citrella Stainton': 89,
  'Toxoptera citricidus': 90,
  'Toxoptera aurantii': 91,
  'Aphis citricola Vander Goot': 92,
  'Scirtothrips dorsalis Hood': 93,
  'Dasineura sp': 94,
  'Lawana imitata Melichar': 95,
  'Salurnis marginella Guerr': 96,
  'Deporaus marginatus Pascoe': 97,
  'Chlumetia transversa': 98,
  'Mango flat beak leafhopper': 99,
  'Rhytidodera bowrinii white': 100,
  'Sternochetus frigidus': 101,
  'Cicadellidae': 102
}  # 修改的地方，修改为自己的类别


# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}

def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.' % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.' % (name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = os.path.splitext(filename)[0]
        return filename
    except:
        raise NotImplementedError('Filename %s is supposed to be an integer.' % (filename))


# xml_list为xml文件存放的txt文件名    xml_dir为真实xml的存放路径    json_file为存放的json路径
def convert(xml_list, xml_dir, json_file):
    list_fp = open(xml_list, 'r')
    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in list_fp:
        line = line.strip()
        line = line + ".xml"
        print("Processing %s" % (line))
        xml_f = os.path.join(xml_dir, line)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s' % (len(path), line))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename, 'height': height, 'width': width,
                 'id': image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            category_id = category
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert (xmax > xmin)
            assert (ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                image_id, 'bbox': [xmin, ymin, o_width, o_height],
                   'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()
    list_fp.close()


if __name__ == '__main__':
    # xml_list为xml文件存放的txt文件名    xml_dir为真实xml的存放路径    json_file为存放的json路径
    # xml_list = 'D:\dataset\ip102\Detection\VOC2007\ImageSets\Main/test.txt'
    xml_list = 'D:\dataset\ip102\Detection\VOC2007\ImageSets\Main/train.txt'
    #xml_list = 'D:\dataset\ip102\Detection\VOC2007\ImageSets\Main/val.txt'
    xml_dir = 'D:\dataset\ip102\Detection\VOC2007\Annotations'
    # json_dir = './data/COCO/annotations/test.json'  # 注意！！！这里test.json先要自己创建，不然
    json_dir = 'D:\dataset\ip102\Detection\COCO/annotations/voc07_train.json'  # 注意！！！这里test.json先要自己创建，不然
    # json_dir = './data/COCO/annotations/val.json'  # 注意！！！这里test.json先要自己创建，不然	       #程序回报权限不足
    convert(xml_list, xml_dir, json_dir)
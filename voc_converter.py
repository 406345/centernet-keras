import json
import os

#############################
#
# 修改这里
#
#############################
ANNOTATION_JSON_FILE = R'G:\data\dp_training\tt100k_2021\annotations_all.json'
DATA_ROOT = R'G:\data\dp_training\tt100k_2021'
TRAIN_FILE_OUTPUT = R'./train.txt'


def build_xml(dict):
    voc_ann_dir = DATA_ROOT + "/VOCDevkit/Annotations"
    voc_img_dir = DATA_ROOT + "/VOCDevkit/JPEGImages"
    os.makedirs(voc_ann_dir, exist_ok=True)
    os.makedirs(voc_img_dir, exist_ok=True)

    for idx, key in enumerate(dict.keys()):
        img = dict[key]
        img_path = DATA_ROOT + '/' + img['path']
        if not os.path.exists(img_path):
            continue

        print('processing xml %s %d/%d\r' % (key, idx + 1, len(dict.keys())))
        img = dict[key]
        dir = img['path'].split('/')[0]
        img_name = img['path'].split('/')[1]

        xml_file = open(voc_ann_dir + '/' + key + '.xml', 'w+')
        xml_template = '''<annotation>
                <folder>{0}</folder>
                <filename>{1}</filename>
                <source>
                    <database>CCTSDB</database>
                    <annotation>CCTSDB</annotation>
                    <image>flickr</image>
                </source>
                <size>
                    <width>2048</width>
                    <height>2048</height>
                    <depth>3</depth>
                </size>
                <segmented>0</segmented>
            '''.format(DATA_ROOT + '/' + dir, img_name)

        # xml_file.write(xml_template)

        for obj in img['objects']:
            xml_template += ('''
                <object>
            		<name>{0}</name>
            		<pose>Unspecified</pose>
            		<truncated>0</truncated>
            		<difficult>0</difficult>
            		<bndbox>
            			<xmin>{1}</xmin>
            			<ymin>{2}</ymin>
            			<xmax>{3}</xmax>
            			<ymax>{4}</ymax>
                    </bndbox>
                </object>
            '''.format(obj['category'],
                       int(obj['bbox']['xmin']),
                       int(obj['bbox']['ymin']),
                       int(obj['bbox']['xmax']),
                       int(obj['bbox']['ymax'])))
        xml_file.write(xml_template + '</annotation>')
        xml_file.close()


def build_train(categories, dict):
    train_file = open(TRAIN_FILE_OUTPUT, 'w')
    for idx, key in enumerate(dict.keys()):
        img_id = key
        img = dict[key]
        img_path = DATA_ROOT + '/' + img['path']

        if not os.path.exists(img_path):
            continue

        line = img_path
        for obj in img['objects']:
            cid = categories.index(obj['category'])
            if cid > len(categories):
                print(cid)
            line += ' {0},{1},{2},{3},{4}'.format(
                int(obj['bbox']['xmin']),
                int(obj['bbox']['ymin']),
                int(obj['bbox']['xmax']),
                int(obj['bbox']['ymax']),
                cid)
        train_file.write(line + '\n')
    pass


if __name__ == "__main__":
    annotation = json.load(open(ANNOTATION_JSON_FILE, 'r'))
    files_dict = {}

    class_file = open('classes.txt','w')
    for t in annotation['types']:
        class_file.write(t + '\n')
    class_file.close()

    for key in annotation['imgs'].keys():
        img_id = key
        img = annotation['imgs'][key]

        if len(img['objects']) == 0:
            continue

        files_dict[key] = img

    build_xml(files_dict)
    build_train(annotation['types'], files_dict)

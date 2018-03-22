'''Convert VOC PASCAL 2007/2012 xml annotations to a list file.'''

import os
import xml.etree.ElementTree as ET


VOC_LABELS = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)

data_dir = './data/VOCdevkit'
voc2007_dir = 'VOC2007/'
voc2012_dir = 'VOC2012/'

txt2007_dir = voc2007_dir + 'ImageSets/Main/'
xml2007_dir = voc2007_dir + 'Annotations/'
jpg2007_dir = voc2007_dir + 'JPEGImages/'
txt2012_dir = voc2012_dir + 'ImageSets/Main/'
xml2012_dir = voc2012_dir + 'Annotations/'
jpg2012_dir = voc2012_dir + 'JPEGImages/'

#################################################################
# create voc0712_trainval.txt file
#################################################################
f = open('voc0712_trainval.txt', 'w')

trainval2007f = open(data_dir + txt2007_dir + 'trainval.txt', 'r')
for img_name in trainval2007f.readlines():
    print('converting %s in voc2007 trainval' % img_name)
    f.write(jpg2007_dir+img_name+' ')
    xml_name = img_name[:-4]+'.xml'

    tree = ET.parse(os.path.join(data_dir, xml2007_dir, xml_name))
    annos = []
    for child in tree.getroot():
        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_label = VOC_LABELS.index(child.find('name').text)
            annos.append('%s %s %s %s %s' % (xmin,ymin,xmax,ymax,class_label))
    f.write('%d %s\n' % (len(annos), ' '.join(annos)))
    
trainval2012f = open(data_dir + txt2012_dir + 'trainval.txt', 'r')
for img_name in trainval2012f.readlines():
    print('converting %s in voc2012 trainval' % img_name)
    f.write(jpg2012_dir+img_name+' ')
    xml_name = img_name[:-4]+'.xml'

    tree = ET.parse(os.path.join(data_dir, xml2012_dir, xml_name))
    annos = []
    for child in tree.getroot():
        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_label = VOC_LABELS.index(child.find('name').text)
            annos.append('%s %s %s %s %s' % (xmin,ymin,xmax,ymax,class_label))
    f.write('%d %s\n' % (len(annos), ' '.join(annos)))
    
f.close()

#################################################################
# create voc2007_test.txt file
#################################################################
f = open('voc2007_test.txt', 'w')

test2007f = open(data_dir + txt2007_dir + 'test.txt', 'r')
for img_name in test2007f.readlines():
    print('converting %s in voc2007 test' % img_name)
    f.write(jpg2007_dir+img_name+' ')
    xml_name = img_name[:-4]+'.xml'

    tree = ET.parse(os.path.join(data_dir, xml2007_dir, xml_name))
    annos = []
    for child in tree.getroot():
        if child.tag == 'object':
            bbox = child.find('bndbox')
            xmin = bbox.find('xmin').text
            ymin = bbox.find('ymin').text
            xmax = bbox.find('xmax').text
            ymax = bbox.find('ymax').text
            class_label = VOC_LABELS.index(child.find('name').text)
            annos.append('%s %s %s %s %s' % (xmin,ymin,xmax,ymax,class_label))
    f.write('%d %s\n' % (len(annos), ' '.join(annos)))
    
f.close()

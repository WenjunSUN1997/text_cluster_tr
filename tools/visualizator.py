from xml_reader import XmlProcessor
from PIL import Image, ImageDraw
import random

def generate_colors(n):
    colors = []
    for i in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
    return colors

def get_gt(xml_file_path):
    annotations = XmlProcessor(1, xml_file_path).get_annotation()
    kind_list = [-1] * len(annotations)
    bbox_list = []
    class_num = 0

    for index_1 in range(len(annotations)):
        bbox_list.append(annotations[index_1]['bbox'][0]+
                          annotations[index_1]['bbox'][2])
        if kind_list[index_1] != -1:
            continue
        kind_list[index_1] = class_num

        for index_2 in range(index_1+1, len(annotations)):
            if kind_list[index_2] != -1:
                continue
            if annotations[index_2]['reading_order'] == \
                    annotations[index_1]['reading_order']:
                kind_list[index_2] = class_num
        class_num += 1

    return kind_list, bbox_list

def draw(img, kind_list, bbox_list, colors):
    draw = ImageDraw.Draw(img)
    for index, bbox in enumerate(bbox_list):
        print(bbox)
        draw.rectangle(bbox, fill=colors[kind_list[index]]+(50,), outline='red')

    img.show()
    return 0

if __name__ == "__main__":
    path = '../AS_TrainingSet_NLF_NewsEye_v2/576450_0003_23676281.xml'
    colors = generate_colors(100)
    kind_list, bbox_list = get_gt(path)
    img = Image.open(path.replace('xml', 'jpg')).convert('RGB')
    img.show()
    draw(img, kind_list, bbox_list, colors)


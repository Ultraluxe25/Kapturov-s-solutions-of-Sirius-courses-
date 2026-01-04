import os
import glob
import xmltodict
from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class FruitDataset(Dataset):
    def __init__(self, data_dir, class2tag, transforms=None, device='cuda', S=7, B=2, C=3):
        """
        :param: S - на сколько ячеек мы разбиваем картинку по ширине и высоте (всего S x S ячеек)
        :param: B - сколько предсказаний (боксов) предсказывается для ячейки
        :param: С - сколько классов
        """
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.box_paths = sorted(glob.glob(os.path.join(data_dir, "*.xml")))
        self.transforms = transforms
        self.device = device
        self.S = S
        self.B = B
        self.C = C
        self.class2tag = class2tag
        assert len(self.image_paths) == len(self.box_paths)

    # Координаты прямоугольников советуем вернуть именно в формате (x_center, y_center, width, height)
    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        boxes, class_labels = self.__get_boxes_from_xml(self.box_paths[idx])

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, class_labels=class_labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        image = torch.from_numpy(image).type(torch.float32)
        # надо привести все данные картинки (сама картинка, боксы и классы, относящиеся к этим боксам)
        # к размеру (B*5 + C)

        target_tensor = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        len_grid_x = image.shape[0] // self.S    # длина ячейки по х
        len_grid_y = image.shape[1] // self.S    # длина ячейки по у

        for index in range(len(boxes)):
            bb = boxes[index]
            label = class_labels[index]
            # достанем координаты центра в картинки нового размера (после ресайза)
            real_center_x = bb[0] * image.shape[0]
            real_center_y = bb[1] * image.shape[1]
            # посчитаем индексы ответственной за текущей бокс ячейки
            grid_x = int(real_center_x // len_grid_x)
            grid_y = int(real_center_y // len_grid_y)
            for i in range(0, self.B * 5, 5):
                if target_tensor[grid_x][grid_y][i + 4] == 0:   # если в текущей пятерке нет бокса
                    target_tensor[grid_x][grid_y][i] = bb[0]    # то кладем шейпы в четверку
                    target_tensor[grid_x][grid_y][i + 1] = bb[1]
                    target_tensor[grid_x][grid_y][i + 2] = bb[2]
                    target_tensor[grid_x][grid_y][i + 3] = bb[3]
                    # конфиденс объектов на трейне, а также вероятность текущего класса равны 1
                    target_tensor[grid_x][grid_y][i + 4] = 1
                    target_tensor[grid_x][grid_y][5 * self.B + label - 1] = 1
                    break
        image = image.to(self.device)
        target_tensor = target_tensor.to(self.device)
        return image, target_tensor

    def __len__(self):
        return len(self.image_paths)

    def __convert_to_yolo_box_params(self, box_coordinates: Tuple[int], im_w, im_h):
        """
        Перейти от [xmin, ymin, xmax, ymax] к [x_center, y_center, width, height].

        Обратите внимание, что параметры [x_center, y_center, width, height] - это
        относительные значение в отрезке [0, 1] относительно исходной картинки

        :param: box_coordinates - координаты коробки в формате [xmin, ymin, xmax, ymax]
        :param: im_w - ширина исходного изображения
        :param: im_h - высота исходного изображения

        :return: координаты коробки в формате [x_center, y_center, width, height]
        """
        ans = list([0] * 4)

        ans[0] = (box_coordinates[0] + box_coordinates[2]) / 2 / im_w  # x_center
        ans[1] = (box_coordinates[1] + box_coordinates[3]) / 2 / im_h  # y_center

        ans[2] = (box_coordinates[2] - box_coordinates[0]) / im_w      # width
        ans[3] = (box_coordinates[3] - box_coordinates[1]) / im_h      # height
        return ans

    def __get_boxes_from_xml(self, xml_filename: str):
        """
        Метод, который считает и распарсит (с помощью xmltodict) переданный xml
        файл и вернет координаты прямоугольников обьектов на соответствующей фотографии
        и название класса обьекта в каждом прямоугольнике

        Обратите внимание, что обьектов может быть как несколько, так и один единственный
        """
        boxes = []
        class_labels = []
        data = open(xml_filename, "r").read()
        dct = xmltodict.parse(data)

        filepath = "data/" + '/'.join(dct['annotation']['path'].split('\\')[-2:])
        im_w, im_h = Image.open(filepath).size

        if type(dct["annotation"]["object"]) is dict:
            target = dct["annotation"]["object"]["name"]
            class_labels.append(self.class2tag[target])
            box = dct["annotation"]["object"]["bndbox"]
            box = list(map(lambda x: int(x), box.values()))
            box = self.__convert_to_yolo_box_params(box, im_w, im_h)
            boxes.append(box)
        else:
            for elem in dct["annotation"]["object"]:
                target = elem["name"]
                class_labels.append(self.class2tag[target])
                box = elem["bndbox"]
                box = list(map(lambda x: int(x), box.values()))     # xmin, ymin, xmax, ymax
                box = self.__convert_to_yolo_box_params(box, im_w, im_h)
                boxes.append(box)

        return boxes, class_labels
import os
from glob import glob
import sys
import numpy as np
from PIL import UnidentifiedImageError
from PIL import Image


class DataLoader():
    def __init__(self, datapath, img_res=(128, 128), scale=4):
        """
        :param string dataset: dataset path 
        :param tuple(int, int) img_res: high resolution images resolution (width, height) 
        :param int scale: low resolution images upscaling factor 
        """

        self.datapath = datapath
        self.img_res = img_res
        self.scale = scale

    @staticmethod
    def read_img(path):
        """ Read image from given path """
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)

    @staticmethod
    def resize_img(image, size, method=Image.BICUBIC):
        """ Resize image to defined size using selected method """
        img = Image.fromarray(image.astype(np.uint8))
        try:
            return np.array(img.resize(size=size, resample=method))
        except ValueError:
            print(f">>>Error occured: unknown resampling filter ({method}), use: \nImage.NEAREST (0), \
            Image.LANCZOS (1), Image.BILINEAR (2), Image.BICUBIC (3), Image.BOX (4) or Image.HAMMING (5)<<<")
            sys.exit(1)

    @staticmethod
    def scale_img(image):
        """ Scale image """
        return image / 127.5 - 1

    @staticmethod
    def unscale_img(image):
        """ Unscale image """
        return (image + 1) * 127.5

    def load_data(self, batch_size=1, testing_data=False):
        """
        :param int batch_size: size of the training batch
        :param bool testing_data: is loaded data a testing data
        :return: two numpy arrays with high resolution images and low resolution images 
        """

        img_types = ('*.jpg', '*.jpeg', '*.png')
        img_paths = []

        for img_type in img_types:
            img_paths.extend(glob(os.path.join(self.datapath, img_type)))

        try:
            batch_images = np.random.choice(img_paths, size=batch_size)
        except ValueError:
            print(f">>>Error occured: dictionary '{self.datapath}' (1) does not exist (2) is empty (3) \
            does not contain data in the specified formats<<<")
            # raise
            sys.exit(1)

        hr_imgs = []
        lr_imgs = []

        # Image reading and resizing
        for img_path in batch_images:
            try:
                img = self.read_img(img_path)
            except UnidentifiedImageError:
                print("Wrong data format")

            low_img_res = (
                int((self.img_res[0] / self.scale)), int((self.img_res[1] / self.scale)))

            hr_img = self.resize_img(img, size=self.img_res)
            lr_img = self.resize_img(img, size=low_img_res)

            # Random flip in the left/right direction for training imgs
            if not testing_data and np.random.random_sample() < 0.3:
                hr_img = np.fliplr(hr_img)
                lr_img = np.fliplr(lr_img)

            hr_imgs.append(hr_img)
            lr_imgs.append(lr_img)

        # Converting the pixel values to a range of between -1 to 1, because generator network has tanh at the end of the network
        hr_imgs = np.array(hr_imgs) / 127.5 - 1.
        lr_imgs = np.array(lr_imgs) / 127.5 - 1.

        return hr_imgs, lr_imgs

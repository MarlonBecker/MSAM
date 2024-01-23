import random
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageOps
import warnings

from utility.args import Args


class AutoAugment(object):
    def __init__(self, policies = None, datasetName = None):
        if policies is not None:
            if datasetName is not None:
                warnings.warn(f"'datasetName' and 'policies' argument were passed to AutoAugment init. 'datasetName will be ignored'")
        else:
            if datasetName is None:
                policies = "sam"
            elif datasetName in ["CIFAR10","CIFAR100"]:
                policies = "sam"
            elif datasetName in ["ImageNet"]:
                policies = "original"
            else:
                raise RuntimeError(f"No default AutoAugment policy given for datasetName '{datasetName}'.")
                
        if policies == "original":
            # original paper policies
            self.policies = [
                ['Invert', 0.1, 7, 'Contrast', 0.2, 6],
                ['Rotate', 0.7, 2, 'TranslateX', 0.3, 9],
                ['Sharpness', 0.8, 1, 'Sharpness', 0.9, 3],
                ['ShearY', 0.5, 8, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.5, 8, 'Equalize', 0.9, 2],
                ['ShearY', 0.2, 7, 'Posterize', 0.3, 7],
                ['Color', 0.4, 3, 'Brightness', 0.6, 7],
                ['Sharpness', 0.3, 9, 'Brightness', 0.7, 9],
                ['Equalize', 0.6, 5, 'Equalize', 0.5, 1],
                ['Contrast', 0.6, 7, 'Sharpness', 0.6, 5],
                ['Color', 0.7, 7, 'TranslateX', 0.5, 8],
                ['Equalize', 0.3, 7, 'AutoContrast', 0.4, 8],
                ['TranslateY', 0.4, 3, 'Sharpness', 0.2, 6],
                ['Brightness', 0.9, 6, 'Color', 0.2, 8],
                ['Solarize', 0.5, 2, 'Invert', 0.0, 3],
                ['Equalize', 0.2, 0, 'AutoContrast', 0.6, 0],
                ['Equalize', 0.2, 8, 'Equalize', 0.6, 4],
                ['Color', 0.9, 9, 'Equalize', 0.6, 6],
                ['AutoContrast', 0.8, 4, 'Solarize', 0.2, 8],
                ['Brightness', 0.1, 3, 'Color', 0.7, 0],
                ['Solarize', 0.4, 5, 'AutoContrast', 0.9, 3],
                ['TranslateY', 0.9, 9, 'TranslateY', 0.7, 9],
                ['AutoContrast', 0.9, 2, 'Solarize', 0.8, 3],
                ['Equalize', 0.8, 8, 'Invert', 0.1, 3],
                ['TranslateY', 0.7, 9, 'AutoContrast', 0.9, 1],
            ]
        elif policies == "sam":
            ### SAM policies ### https://github.com/google-research/sam/blob/main/autoaugment/policies.py
            exp0_0 = [
                [('Invert', 0.1, 7), ('Contrast', 0.2, 6)],
                [('Rotate', 0.7, 2), ('TranslateX', 0.3, 9)],
                [('Sharpness', 0.8, 1), ('Sharpness', 0.9, 3)],
                [('ShearY', 0.5, 8), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.5, 8), ('Equalize', 0.9, 2)]]
            exp0_1 = [
                [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 3)],
                [('TranslateY', 0.9, 9), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.9, 2), ('Solarize', 0.8, 3)],
                [('Equalize', 0.8, 8), ('Invert', 0.1, 3)],
                [('TranslateY', 0.7, 9), ('AutoContrast', 0.9, 1)]]
            exp0_2 = [
                [('Solarize', 0.4, 5), ('AutoContrast', 0.0, 2)],
                [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)],
                [('AutoContrast', 0.9, 0), ('Solarize', 0.4, 3)],
                [('Equalize', 0.7, 5), ('Invert', 0.1, 3)],
                [('TranslateY', 0.7, 9), ('TranslateY', 0.7, 9)]]
            exp0_3 = [
                [('Solarize', 0.4, 5), ('AutoContrast', 0.9, 1)],
                [('TranslateY', 0.8, 9), ('TranslateY', 0.9, 9)],
                [('AutoContrast', 0.8, 0), ('TranslateY', 0.7, 9)],
                [('TranslateY', 0.2, 7), ('Color', 0.9, 6)],
                [('Equalize', 0.7, 6), ('Color', 0.4, 9)]]
            exp1_0 = [
                [('ShearY', 0.2, 7), ('Posterize', 0.3, 7)],
                [('Color', 0.4, 3), ('Brightness', 0.6, 7)],
                [('Sharpness', 0.3, 9), ('Brightness', 0.7, 9)],
                [('Equalize', 0.6, 5), ('Equalize', 0.5, 1)],
                [('Contrast', 0.6, 7), ('Sharpness', 0.6, 5)]]
            exp1_1 = [
                [('Brightness', 0.3, 7), ('AutoContrast', 0.5, 8)],
                [('AutoContrast', 0.9, 4), ('AutoContrast', 0.5, 6)],
                [('Solarize', 0.3, 5), ('Equalize', 0.6, 5)],
                [('TranslateY', 0.2, 4), ('Sharpness', 0.3, 3)],
                [('Brightness', 0.0, 8), ('Color', 0.8, 8)]]
            exp1_2 = [
                [('Solarize', 0.2, 6), ('Color', 0.8, 6)],
                [('Solarize', 0.2, 6), ('AutoContrast', 0.8, 1)],
                [('Solarize', 0.4, 1), ('Equalize', 0.6, 5)],
                [('Brightness', 0.0, 0), ('Solarize', 0.5, 2)],
                [('AutoContrast', 0.9, 5), ('Brightness', 0.5, 3)]]
            exp1_3 = [
                [('Contrast', 0.7, 5), ('Brightness', 0.0, 2)],
                [('Solarize', 0.2, 8), ('Solarize', 0.1, 5)],
                [('Contrast', 0.5, 1), ('TranslateY', 0.2, 9)],
                [('AutoContrast', 0.6, 5), ('TranslateY', 0.0, 9)],
                [('AutoContrast', 0.9, 4), ('Equalize', 0.8, 4)]]
            exp1_4 = [
                [('Brightness', 0.0, 7), ('Equalize', 0.4, 7)],
                [('Solarize', 0.2, 5), ('Equalize', 0.7, 5)],
                [('Equalize', 0.6, 8), ('Color', 0.6, 2)],
                [('Color', 0.3, 7), ('Color', 0.2, 4)],
                [('AutoContrast', 0.5, 2), ('Solarize', 0.7, 2)]]
            exp1_5 = [
                [('AutoContrast', 0.2, 0), ('Equalize', 0.1, 0)],
                [('ShearY', 0.6, 5), ('Equalize', 0.6, 5)],
                [('Brightness', 0.9, 3), ('AutoContrast', 0.4, 1)],
                [('Equalize', 0.8, 8), ('Equalize', 0.7, 7)],
                [('Equalize', 0.7, 7), ('Solarize', 0.5, 0)]]
            exp1_6 = [
                [('Equalize', 0.8, 4), ('TranslateY', 0.8, 9)],
                [('TranslateY', 0.8, 9), ('TranslateY', 0.6, 9)],
                [('TranslateY', 0.9, 0), ('TranslateY', 0.5, 9)],
                [('AutoContrast', 0.5, 3), ('Solarize', 0.3, 4)],
                [('Solarize', 0.5, 3), ('Equalize', 0.4, 4)]]
            exp2_0 = [
                [('Color', 0.7, 7), ('TranslateX', 0.5, 8)],
                [('Equalize', 0.3, 7), ('AutoContrast', 0.4, 8)],
                [('TranslateY', 0.4, 3), ('Sharpness', 0.2, 6)],
                [('Brightness', 0.9, 6), ('Color', 0.2, 8)],
                [('Solarize', 0.5, 2), ('Invert', 0.0, 3)]]
            exp2_1 = [
                [('AutoContrast', 0.1, 5), ('Brightness', 0.0, 0)],
                [('Cutout', 0.2, 4), ('Equalize', 0.1, 1)],
                [('Equalize', 0.7, 7), ('AutoContrast', 0.6, 4)],
                [('Color', 0.1, 8), ('ShearY', 0.2, 3)],
                [('ShearY', 0.4, 2), ('Rotate', 0.7, 0)]]
            exp2_2 = [
                [('ShearY', 0.1, 3), ('AutoContrast', 0.9, 5)],
                [('TranslateY', 0.3, 6), ('Cutout', 0.3, 3)],
                [('Equalize', 0.5, 0), ('Solarize', 0.6, 6)],
                [('AutoContrast', 0.3, 5), ('Rotate', 0.2, 7)],
                [('Equalize', 0.8, 2), ('Invert', 0.4, 0)]]
            exp2_3 = [
                [('Equalize', 0.9, 5), ('Color', 0.7, 0)],
                [('Equalize', 0.1, 1), ('ShearY', 0.1, 3)],
                [('AutoContrast', 0.7, 3), ('Equalize', 0.7, 0)],
                [('Brightness', 0.5, 1), ('Contrast', 0.1, 7)],
                [('Contrast', 0.1, 4), ('Solarize', 0.6, 5)]]
            exp2_4 = [
                [('Solarize', 0.2, 3), ('ShearX', 0.0, 0)],
                [('TranslateX', 0.3, 0), ('TranslateX', 0.6, 0)],
                [('Equalize', 0.5, 9), ('TranslateY', 0.6, 7)],
                [('ShearX', 0.1, 0), ('Sharpness', 0.5, 1)],
                [('Equalize', 0.8, 6), ('Invert', 0.3, 6)]]
            exp2_5 = [
                [('AutoContrast', 0.3, 9), ('Cutout', 0.5, 3)],
                [('ShearX', 0.4, 4), ('AutoContrast', 0.9, 2)],
                [('ShearX', 0.0, 3), ('Posterize', 0.0, 3)],
                [('Solarize', 0.4, 3), ('Color', 0.2, 4)],
                [('Equalize', 0.1, 4), ('Equalize', 0.7, 6)]]
            exp2_6 = [
                [('Equalize', 0.3, 8), ('AutoContrast', 0.4, 3)],
                [('Solarize', 0.6, 4), ('AutoContrast', 0.7, 6)],
                [('AutoContrast', 0.2, 9), ('Brightness', 0.4, 8)],
                [('Equalize', 0.1, 0), ('Equalize', 0.0, 6)],
                [('Equalize', 0.8, 4), ('Equalize', 0.0, 4)]]
            exp2_7 = [
                [('Equalize', 0.5, 5), ('AutoContrast', 0.1, 2)],
                [('Solarize', 0.5, 5), ('AutoContrast', 0.9, 5)],
                [('AutoContrast', 0.6, 1), ('AutoContrast', 0.7, 8)],
                [('Equalize', 0.2, 0), ('AutoContrast', 0.1, 2)],
                [('Equalize', 0.6, 9), ('Equalize', 0.4, 4)]]
            exp0s = exp0_0 + exp0_1 + exp0_2 + exp0_3
            exp1s = exp1_0 + exp1_1 + exp1_2 + exp1_3 + exp1_4 + exp1_5 + exp1_6
            exp2s = exp2_0 + exp2_1 + exp2_2 + exp2_3 + exp2_4 + exp2_5 + exp2_6 + exp2_7
            policies = exp0s + exp1s + exp2s

            self.policies = [[operation for exp in pol for operation in exp] for pol in policies]
        else:
            raise RuntimeError(f"Policies '{policies}' unknown.")


    def __call__(self, img):
        img = apply_policy(img, self.policies[random.randrange(len(self.policies))])
        return img


operations = {
    'ShearX': lambda img, magnitude: shear_x(img, magnitude),
    'ShearY': lambda img, magnitude: shear_y(img, magnitude),
    'TranslateX': lambda img, magnitude: translate_x(img, magnitude),
    'TranslateY': lambda img, magnitude: translate_y(img, magnitude),
    'Rotate': lambda img, magnitude: rotate(img, magnitude),
    'AutoContrast': lambda img, magnitude: auto_contrast(img, magnitude),
    'Invert': lambda img, magnitude: invert(img, magnitude),
    'Equalize': lambda img, magnitude: equalize(img, magnitude),
    'Solarize': lambda img, magnitude: solarize(img, magnitude),
    'Posterize': lambda img, magnitude: posterize(img, magnitude),
    'Contrast': lambda img, magnitude: contrast(img, magnitude),
    'Color': lambda img, magnitude: color(img, magnitude),
    'Brightness': lambda img, magnitude: brightness(img, magnitude),
    'Sharpness': lambda img, magnitude: sharpness(img, magnitude),
    'Cutout': lambda img, magnitude: cutout(img, magnitude),
}


def apply_policy(img, policy):
    if random.random() < policy[1]:
        img = operations[policy[0]](img, policy[2])
    if random.random() < policy[4]:
        img = operations[policy[3]](img, policy[5])

    return img


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = offset_matrix @ matrix @ reset_matrix
    return transform_matrix


def shear_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def shear_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-0.3, 0.3, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]), 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_x(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, 0],
                                 [0, 1, img.shape[1]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def translate_y(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-150/331, 150/331, 11)

    transform_matrix = np.array([[1, 0, img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def rotate(img, magnitude):
    img = np.array(img)
    magnitudes = np.linspace(-30, 30, 11)
    theta = np.deg2rad(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                 [np.sin(theta), np.cos(theta), 0],
                                 [0, 0, 1]])
    transform_matrix = transform_matrix_offset_center(transform_matrix, img.shape[0], img.shape[1])
    affine_matrix = transform_matrix[:2, :2]
    offset = transform_matrix[:2, 2]
    img = np.stack([ndimage.interpolation.affine_transform(
                    img[:, :, c],
                    affine_matrix,
                    offset) for c in range(img.shape[2])], axis=2)
    img = Image.fromarray(img)
    return img


def auto_contrast(img, magnitude):
    img = ImageOps.autocontrast(img)
    return img


def invert(img, magnitude):
    img = ImageOps.invert(img)
    return img


def equalize(img, magnitude):
    img = ImageOps.equalize(img)
    return img


def solarize(img, magnitude):
    magnitudes = np.linspace(0, 256, 11)
    img = ImageOps.solarize(img, random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def posterize(img, magnitude):
    magnitudes = np.linspace(4, 8, 11)
    img = ImageOps.posterize(img, int(round(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))))
    return img


def contrast(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Contrast(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def color(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Color(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def brightness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def sharpness(img, magnitude):
    magnitudes = np.linspace(0.1, 1.9, 11)
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(magnitudes[magnitude], magnitudes[magnitude+1]))
    return img


def cutout(org_img, magnitude=None, mask_val=None):

    img = np.copy(org_img)
    if mask_val is None:
        mask_val = [img.mean() / 255] * img.shape[2]


    if magnitude is None:
        mask_size = 16
    else:
        magnitudes = np.linspace(0, 60/331, 11)
        mask_size = int(round(img.shape[0]*random.uniform(magnitudes[magnitude], magnitudes[magnitude+1])))
    top = np.random.randint(0 - mask_size//2, img.shape[0] - mask_size//2)
    left = np.random.randint(0 - mask_size//2, img.shape[1] - mask_size//2)
    bottom = min(img.shape[0], top + mask_size)
    right = min(img.shape[1], left + mask_size)

    for i in range(img.shape[2]):
        img[max(0, top):bottom, max(0, left):right, i].fill(mask_val[i] * 255)

    img = Image.fromarray(img)

    return img


    
class Cutout:
    def __init__(self, mask_val):
        assert Args.cutoutProp <= 1 and Args.cutoutProp >= 0

        self.p = Args.cutoutProp
        self.mask_val = mask_val

    def __call__(self, image):
        if np.random.random() > self.p:
            return image
        return cutout(image, mask_val = self.mask_val)

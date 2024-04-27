import torchio as tio
import torch
import torch.nn.functional as F
import random

class Rotation(tio.transforms.Transform):

    '''
    Note: tiotransforms.RandomAffine changes the datatype of images and labels to torch.float32, but we want to keep the input type
    '''

    def __init__(self, p: float = 0.2):
        super().__init__(p=p)
        self.T = tio.transforms.RandomAffine(scales=0, degrees=30, default_pad_value='otsu', image_interpolation='bspline', p=1)

    def apply_transform(self, subject):

        data_types = {}

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                data_types[key] = value.data.dtype

        subject = self.T(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                value.set_data(value.data.to(data_types[key]))

        return subject
    

class Scaling(tio.transforms.Transform):

    '''
    Note: tiotransforms.RandomAffine changes the datatype of images and labels to torch.float32, but we want to keep the input type
    '''

    def __init__(self, p: float = 0.2):
        super().__init__(p=p)
        self.T = tio.transforms.RandomAffine(scales=(0.7, 1.4), degrees=0, isotropic=True, default_pad_value='otsu', image_interpolation='bspline', p=1)
    
    def apply_transform(self, subject):

        data_types = {}

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                data_types[key] = value.data.dtype

        subject = self.T(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                value.set_data(value.data.to(data_types[key]))

        return subject
    

class GaussianNoise(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)
        self.T = tio.transforms.RandomNoise(mean=0, std=(0, 0.316), p=1)
    
    def apply_transform(self, subject):
        return self.T(subject)
    

class GaussianBlur(tio.transforms.Transform):
    
        def __init__(self, p: float = 0.1):
            super().__init__(p=p)
            self.T = tio.transforms.RandomBlur(std=(0.5, 1.5), p=1)
    
        def apply_transform(self, subject):
            return self.T(subject)


class Brightness(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply_transform(self, subject):

        x = random.uniform(0.7, 1.3)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                value.set_data(value.data * x)

        return subject
    

class Contrast(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply_transform(self, subject):

        x = random.uniform(0.65, 1.5)

        min_max = self.get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                scaled_data = value.data * x
                clamped_data = torch.clamp(scaled_data, min_max[key][0], min_max[key][1])
                value.set_data(clamped_data)

        return subject
    
    def get_min_max(self, subject):

        min_max = {}
        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
                min_max[key] = (min, max)
        
        return min_max


class SimulateLowResolution(tio.transforms.Transform):

    def __init__(self, p: float = 0.25):
        super().__init__(p=p)

    def apply_transform(self, subject):

        for image in subject.get_images():
            T = tio.transforms.RandomAnisotropy(axes=(0,1,2), downsampling=(1,2), image_interpolation="bspline", p=0.5)
            image.set_data(T(image.data))

        return subject


class Gamma(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply_transform(self, subject):

        gamma = random.uniform(0.7, 1.5)

        min_max = self.get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min, max = min_max[key][0], min_max[key][1]
                normalised_data = (value.data - min) / (max - min)
                if random.random() < 0.15:
                    augmented_data = 1 - (1 - normalised_data) ** gamma
                else:
                    augmented_data = normalised_data ** gamma
                rescaled_data = augmented_data * (max - min) + min
                value.set_data(rescaled_data)

        return subject

    def get_min_max(self, subject):

        min_max = {}
        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
                min_max[key] = (min, max)
        
        return min_max
    

class Mirroring(tio.transforms.Transform):

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)
        self.T = tio.transforms.RandomFlip(axes=(0,1,2), flip_probability=0.5)

    def apply_transform(self, subject):
        return self.T(subject)
      
      
class ComposednnUNetTransforms(tio.transforms.Transform):

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

        T_list = [
            Rotation(),
            Scaling(),
            Mirroring()
            GaussianNoise(),
            GaussianBlur(),
            Brightness(),
            Contrast(),
            SimulateLowResolution(),
            Gamma()
            ]

        self.T = tio.transforms.Compose(T_list)

    def apply_transform(self, subject):
        return self.T(subject)
        

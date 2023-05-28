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

    def apply_transform(self, subject):

        data_types = {}

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                data_types[key] = value.data.dtype

        T = tio.transforms.RandomAffine(scales=0, degrees=30, default_pad_value='otsu', image_interpolation='bspline', p=1)

        subject = T(subject)

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

    def apply_transform(self, subject):

        scale = random.uniform(0.7, 1.4)

        data_types = {}

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                data_types[key] = value.data.dtype

        T = tio.transforms.RandomAffine(scales=(scale, scale, scale, scale, scale, scale), degrees=0, default_pad_value='otsu', image_interpolation='bspline', p=1)

        subject = T(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage) or isinstance(value, tio.LabelMap):
                value.set_data(value.data.to(data_types[key]))

        return subject
    

class GaussianNoise(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply_transform(self, subject):

        T = tio.transforms.RandomNoise(mean=0, std=(0, 0.316), p=1)

        return T(subject)
    

class GaussianBlur(tio.transforms.Transform):
    
        def __init__(self, p: float = 0.1):
            super().__init__(p=p)
    
        def apply_transform(self, subject):
    
            T = tio.transforms.RandomBlur(std=(0.5, 1.5), p=1)
    
            return T(subject)


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

        min, max = self.get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                scaled_data = value.data * x
                clamped_data = torch.clamp(scaled_data, min, max)
                value.set_data(clamped_data)

        return subject
    
    def get_min_max(self, subject):

        min, max = 0, 0

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
        
        return min, max
    

class SimulateLowResolution(tio.transforms.Transform):

    def __init__(self, p: float = 0.125):
        super().__init__(p=p)

    def apply_transform(self, subject):

        spacing = random.uniform(1, 2)

        input_spatial_shape = subject.spatial_shape
        T_shape = tio.transforms.CropOrPad(input_spatial_shape)

        T_down = tio.transforms.Resample(
            target=spacing,
            image_interpolation='nearest',
            scalars_only=True,
            p=1,
        )

        T_up = tio.transforms.Resample(
            target=1,
            image_interpolation='bspline',
            scalars_only=True,
            p=1
        )

        low_res_subject = T_up(T_down(subject))

        for image in low_res_subject.get_images():
            if image.spatial_shape != input_spatial_shape:
                image.set_data(T_shape(image.data))

        return low_res_subject
    

class Gamma(tio.transforms.Transform):

    def __init__(self, p: float = 0.15):
        super().__init__(p=p)

    def apply_transform(self, subject):

        gamma = random.uniform(0.7, 1.5)

        min, max = self.get_min_max(subject)

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                normalised_data = (value.data - min) / (max - min)
                if random.random() < 0.15:
                    augmented_data = 1 - (1 - normalised_data) ** gamma
                else:
                    augmented_data = normalised_data ** gamma
                rescaled_data = augmented_data * (max - min) + min
                value.set_data(rescaled_data)

        return subject

    def get_min_max(self, subject):

        min, max = 0, 0

        for key, value in subject.items():
            if isinstance(value, tio.ScalarImage):
                min = value.data.min()
                max = value.data.max()
        
        return min, max
    

class Mirroring(tio.transforms.Transform):

    def __init__(self, p: float = 0.5):
        super().__init__(p=p)

    def apply_transform(self, subject):

        T_list = [tio.transforms.RandomFlip(axes=i, flip_probability=1) for i in range(3) if random.random() < 0.5]

        T_composed = tio.transforms.Compose(T_list)

        return T_composed(subject)
      
      
class ComposednnUNetTransforms(tio.transforms.Transform):

    def apply_transform(self, subject):

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

        T_composed = tio.transforms.Compose(T_list)

        return T_composed(subject)
        

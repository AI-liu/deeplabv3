import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
from torchvision import transforms



def remap(image, old_values, new_values):
    assert isinstance(image, Image.Image) or isinstance(
        image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
    assert type(new_values) is tuple, "new_values must be of type tuple"
    assert type(old_values) is tuple, "old_values must be of type tuple"
    assert len(new_values) == len(
        old_values), "new_values and old_values must have the same length"

    # If image is a PIL.Image convert it to a numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Replace old values by the new ones
    tmp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        # Since tmp is already initialized as zeros we can skip new values
        # equal to 0
        if new != 0:
            tmp[image == old] = new

    return Image.fromarray(tmp)




class PILToLongTensor(object):
    """Converts a ``PIL Image`` to a ``torch.LongTensor``.

    Code adapted from: http://pytorch.org/docs/master/torchvision/transforms.html?highlight=totensor

    """

    def __call__(self, pic):
        """Performs the conversion from a ``PIL Image`` to a ``torch.LongTensor``.

        Keyword arguments:
        - pic (``PIL.Image``): the image to convert to ``torch.LongTensor``

        Returns:
        A ``torch.LongTensor``.

        """
        if not isinstance(pic, Image.Image):
            raise TypeError("pic should be PIL Image. Got {}".format(
                type(pic)))

        # handle numpy array
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.long()

        # Convert PIL image to ByteTensor
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # Reshape tensor
        nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        # Convert to long and squeeze the channels
        return img.transpose(0, 1).transpose(0,
                                             2).contiguous().long().squeeze_()


class CityScapeDataSet(torch.utils.data.Dataset):

    # The values associated with the 35 classes
    full_classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, -1)
    # The values above are remapped to the following
    new_classes = (0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 5, 0, 0, 0, 6, 0, 7,
                   8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 17, 18, 19, 0)


    def __init__(self, data_path='/media//DATA/dataset/cityscapes'):
        self.data_path = data_path
        self.image_path = '/gtFine_trainvaltest/leftImg8bit/train/aachen/'
        self.label_path = '/gtFine_trainvaltest/gtFine/train/aachen/'
        self.file_list = os.listdir(self.data_path + self.image_path)


    def __getitem__(self, i):

        input_image = Image.open(self.data_path + self.image_path + self.file_list[i])
        input_image = input_image.resize((int(input_image.width/2),int(input_image.height/2)),Image.NEAREST)
        preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
        input_tensor = preprocess(input_image)

        label_image = Image.open(self.data_path + self.label_path + self.file_list[i][:-15] + 'gtFine_labelIds.png')
        label_image = label_image.resize((int(label_image.width/2),int(label_image.height/2)),Image.NEAREST)
        # Remap class labels
        label_image = remap(label_image, self.full_classes, self.new_classes)

        label_tensor = PILToLongTensor()(label_image)
        
        return  input_tensor , label_tensor

    def __len__(self):
        return len(self.file_list)

if __name__ == "__main__":
    print("hello dataset")

    batch_size=12
    dataset=CityScapeDataSet()
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True)
    for batch_idx , (input_tensor , label_tensor) in enumerate(data_loader):
        print(input_tensor[1].shape)
        print(label_tensor[1].shape)


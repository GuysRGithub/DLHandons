import os
from AI.MxnetExamples.SSD.Dataset.IMDB import IMdb


class TestDB(IMdb):
    def __init__(self, images, root_dir=None, extension=None):
        if not isinstance(images, list):
            images = [images]
        num_images = [images]
        super(TestDB, self).__init__("test" + str(num_images))
        self.image_set_index = images
        self.num_images = num_images
        self.root_dir = root_dir if root_dir else None
        self.extension = extension if extension else None

    def image_path_from_index(self, index):
        """
           given image index, return full path
           Parameters:
           ----------
           index: int
               index of a specific image
           Returns
           ----------
           path of this image
        """
        name = self.image_set_index[index]
        if self.extension:
            name += self.extension
        if self.root_dir:
            name = os.path.join(self.root_dir, name)
        assert os.path.exists(name), "Path does not exits: {}"\
            .format(name)
        return name

    def label_from_index(self, index):
        return RuntimeError("TestDb does not support label loading")
    


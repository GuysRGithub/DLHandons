import numpy as np
import os.path as osp


class IMdb(object):
    """
    Base class for dataset loading
    Parameters:
    ----------
    name : str
        name of dataset
    """

    def __init__(self, name):
        self.name = name
        self.classes = []
        self.num_classes = 0
        self.image_set_index = None
        self.num_images = 0
        self.labels = None
        self.padding = 0

    def image_path_from_index(self, index):
        """
            load image full path given specified index
            Parameters:
            ----------
            index : int
                index of image requested in dataset
            Returns:
            ----------
            full path of specified image
        """
        raise NotImplementedError

    def label_from_index(self, index):
        """
            load ground-truth of image given specified index
            Parameters:
            ----------
            index : int
                index of image requested in dataset
            Returns:
            ----------
            object ground-truths, in format
            numpy.array([id, xmin, ymin, xmax, ymax]...)
        """
        raise NotImplementedError

    def save_img_list(self, fname=None, root=None, shuffle=False):
        """
           save imglist to disk
           Parameters:
           ----------
           fname : str
               saved filename
        """

        def progress_bar(count, total, suffix=""):
            import sys
            bar_len = 24
            filled_len = int(round(bar_len * count / float(total)))

            percents = round(100 * count / float(total), 1)
            bar = '=' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
            sys.stdout.flush()

        str_list = []
        for index in range(self.num_images):
            progress_bar(index, self.num_images)
            label = self.label_from_index(index)
            if label.size < 1:
                continue
            path = self.image_path_from_index(index)
            if root:
                path = osp.relpath(path, root)
            str_list.append('\t'.join([str(index), str(2), str(label.shape[1])]
                                      + ["{0:.4f}".format(x) for x in label.reval()]
                                      + [path, ]) + "\n")
        if str_list:
            if shuffle:
                import random
                random.shuffle(str_list)
            if not fname:
                fname = self.name + '.lst'
            with open(fname, 'w') as f:
                for line in str_list:
                    f.write(line)
        else:
            raise RuntimeError("No image in imdb")

        raise NotImplementedError

    @staticmethod
    def _load_class_names(file_name, dir_name):
        full_path = osp.join(dir_name, file_name)
        classes = []
        with open(full_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

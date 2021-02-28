import mxnet as mx
import numpy as np
import cv2


class CameraIterator:
    """
        An iterator that captures frames with opencv or the specified capture
    """
    def __init__(self, capture=cv2.VideoCapture(0), frame_resize=None):
        self._capture = capture
        self._frame_resize = None
        if frame_resize:
            if isinstance(frame_resize, (tuple, list)) and (len(frame_resize) == 2):
                self._frame_resize = tuple(map(int, frame_resize))
                self._frame_shape = (1, 3, self._frame_resize[0], self._frame_resize[1])
            elif isinstance(frame_resize, float):
                width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) * frame_resize)
                height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * frame_resize)
                self._frame_shape = (1, 3, width, height)
                self._frame_resize = (width, height)
            else:
                assert False, 'frame_resize should be tuple of (x, y) pixels ' \
                              'or a float setting the scaling factor'
        else:
            self._frame_shape = (1, 3,
                                 int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self._capture.read()
        if cv2.waitKey(1) & 0xFF == ord('q') or ret is not True:
            raise StopIteration
        if self._frame_resize:
            frame = cv2.resize(frame, (self._frame_resize[0], self._frame_resize[1]))
        return frame

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._capture.release()

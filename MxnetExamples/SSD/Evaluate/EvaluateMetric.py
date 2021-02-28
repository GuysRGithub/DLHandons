import numpy as np
import mxnet as mx


class MApMetric(mx.metric.EvalMetric):
    """
        Calculate mean AP for object detection task
        Parameters:
        ---------
        ovp_thresh : float
            overlap threshold for TP
        use_difficult : boolean
            use difficult ground-truths if applicable, otherwise just ignore
        class_names : list of str
            optional, if provided, will print out AP for each class
        pred_idx : int
            prediction index in network output list
    """
    def __init__(self, ovp_thresh=0.5, use_difficult=False,
                 class_names=None, pred_idx=0):
        super(MApMetric, self).__init__(name="mAP")
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), 'must provide names as str'
            num = len(class_names)
            self.name = class_names + ['mAP']
            self.num = num + 1
        self.reset()
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self.records = dict()
        self.counts = dict()

    def get(self):
        """Get the current evaluation result.
           Returns
           -------
           name : str
              Name of the metric.
           value : float
              Value of the evaluation.
        """
        self._update()
        if self.num is None:
            if self.num_inst == 0:
                return self.name, float('nan')
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            names = ['%s' % (self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan')
                      for x, y in zip(self.sum_metric, self.num_inst)]
            return names, values

    def update(self, labels, preds):
        """
            Update internal records. This function now only update internal buffer,
            sum_metric and num_inst are updated in _update() function instead when
            get() is called to return results.
            Params:
            ----------
            labels: mx.nd.array (n * 6) or (n * 5), difficult column is optional
                2-d array of ground-truths, n objects(id-xmin-ymin-xmax-ymax-[difficult])
            preds: mx.nd.array (m * 6)
                2-d array of detections, m objects(id-score-xmin-ymin-xmax-ymax)
        """
        def iou(x, ys):
            """
                Calculate intersection-over-union overlap
                Params:
                ----------
                x : numpy.array
                    single box [xmin, ymin ,xmax, ymax]
                ys : numpy.array
                    multiple box [[xmin, ymin, xmax, ymax], [...], ]
                Returns:
                -----------
                numpy.array
                    [iou1, iou2, ...], size == ys.shape[0]
            """
            ix_min = np.maximum(ys[:, 0], x[0])
            iy_min = np.maximum(ys[:, 1], x[1])
            ix_max = np.minimum(ys[:, 2], x[2])
            iy_max = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ix_max - ix_min, 0.)
            ih = np.maximum(iy_max - iy_min, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                  (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0
            return ious

        for i in range(labels[0].shape[0]):
            label = labels[0][i].asnumpy()
            if np.sum(label[:, 0] >= 0) < 1:
                continue
            pred = preds[self.pred_idx][i].asnumpy()
            while pred.shape[0] > 0:
                cid = int(pred[0, 0])
                indices = np.where(pred[:, 0].astype(int) == cid)[0]
                if cid < 0:
                    pred = np.delete(pred, indices, axis=0)
                    continue
                dets = pred[indices]
                pred = np.delete(pred, indices, axis=0)
                dets = dets[dets[:, 1].argsort()[::-1]]
                records = np.hstack((dets[:, 1][:, np.newaxis], np.zeros((dets.shape[0], 1))))
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                gts = label[label_indices, :]
                label = np.delete(label, label_indices, axis=0)
                if gts.size > 0:
                    found = [False] * gts.shape[0]
                    for j in range(dets.shape[0]):
                        ious = iou(dets[j, 2:], gts[:, 1:5])
                        ov_arg_max = np.argmax(ious)
                        ov_max = ious[ov_arg_max]
                        if ov_max > self.ovp_thresh:
                            if (not self.use_difficult and gts.shape[1] >= 6 and
                            gts[ov_arg_max, 5] > 0):
                                pass
                            else:
                                if not found[ov_arg_max]:
                                    records[j, -1] = 1
                                    found[ov_arg_max] = True
                                else:
                                    records[j, -1] = 2
                        else:
                            records[j, -1] = 2
                else:
                    records[:, -1] = 2






# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from model.Detector.detect_utils import MyRCNN, MyROIHeads
import numpy as np
import torch


class MyPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)

    def __call__(self, original_image, objects=None):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[..., ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda()

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions


class VisualizationDemo(object):
    def __init__(self, detector_cfg, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.cfg = cfg
        self.metadata = MetadataCatalog.get(
            detector_cfg.DATASETS.TEST[0] if len(detector_cfg.DATASETS.TEST) else "__unused"
        )
        self.instance_mode = instance_mode
        self.parallel = parallel
        self.predictor = MyPredictor(detector_cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image[:,])
        objects = predictions['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
        scores = predictions['instances']._fields['scores'].cpu().detach().numpy()
        classes = predictions['instances']._fields['pred_classes'].cpu().detach().numpy()
        segs = predictions['instances']._fields['pred_masks'].cpu().detach().numpy()
        segs_vis = segs
        # visualizer = Visualizer(image[:, :, ::-1], metadata=self.metadata, scale=1.0)
        # vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        # plt.imshow(vis.get_image()[:, :, ::-1]);
        # plt.show()
        del predictions
        H, W, C = image.shape
        if len(objects) > 0:
            objects[:, 0] = np.maximum(0, objects[:, 0])
            objects[:, 1] = np.maximum(0, objects[:, 1])
            objects[:, 2] = np.minimum(W-1, objects[:, 2])
            objects[:, 3] = np.minimum(H-1, objects[:, 3])
            score_idx = np.argsort(-scores)#[:100]
            objects = objects[score_idx]
            scores = scores[score_idx]
            classes = classes[score_idx]
            segs = segs[score_idx]
            segs_vis = segs[scores > 0.5]
            classes_vis = classes[scores > 0.5]
        seg_mask = np.ones((H, W, 80), dtype=np.int32) * (-1)
        for seg_i, seg_ in enumerate(segs_vis):
            seg_mask[...,classes_vis[seg_i]] += seg_
        seg_mask_mask = seg_mask.max(-1) ==-1
        seg_mask = seg_mask.argmax(-1)
        seg_mask[seg_mask_mask] = -1
        return objects, scores, classes, seg_mask

    def combine_edges(self, objects, scores, classes, edges):
        rois_inside = (objects[:, 2] - objects[:, 0] < edges[0]+1) & (objects[:, 2] - objects[:, 0] > 0) & ((objects[:, 2] - objects[:, 0]) * (objects[:, 3] - objects[:, 1]) > 70)
        if np.sum(rois_inside) > 0:
            objects = objects[rois_inside]
            classes = classes[rois_inside]
            scores = scores[rois_inside]
            sorted_idx = np.argsort(-objects[:, 0])
            objects = objects[sorted_idx]
            classes = classes[sorted_idx]
            scores = scores[sorted_idx]
            if len(objects) > 0:
                combined = np.zeros(len(objects))
                dist_2d = objects[:, 2][None] - objects[:, 0][:, None]
                dist_2d[np.eye(len(dist_2d))==1] = -100
                on_edge = (np.abs(np.min(objects[:, 2][None] - np.array(edges)[:, None], 0)) < 3) | (np.abs(np.min(objects[:, 0][None] - np.array(edges)[:, None], 0)) < 3)
                for ii in range(len(objects)):
                    close_bbox_2d = set(np.where((dist_2d[ii] > -20) & (dist_2d[ii] < 20))[0]) #on the right, pixel around +- 20
                    bbox_at_edge = on_edge[ii] #on the edge of multiview images
                    same_category = set(np.where(classes[ii] == classes)[0])
                    close_bbox_idx = close_bbox_2d.intersection(same_category)
                    if len(close_bbox_idx) > 0:
                        close_bbox = objects[list(close_bbox_idx)[0]]
                        objects[ii][0] = np.minimum(objects[ii][0], close_bbox[0])
                        objects[ii][1] = np.minimum(objects[ii][1], close_bbox[1])
                        objects[ii][2] = np.maximum(objects[ii][2], close_bbox[2])
                        objects[ii][3] = np.maximum(objects[ii][3], close_bbox[3])
                        combined[list(close_bbox_idx)[0]] = 1
                objects = objects[~combined.astype(bool)]
                classes = classes[~combined.astype(bool)]
                scores = scores[~combined.astype(bool)]
        return objects, scores, classes

#
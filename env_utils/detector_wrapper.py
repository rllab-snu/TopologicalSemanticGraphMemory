# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
import numpy as np
import torch
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
from detectron2.config import get_cfg
import os
from types import SimpleNamespace
from NuriUtils.ncutils import append_to_dict
from model.Detector.detect_utils import MyRCNN, MyROIHeads

class MyPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        # self.model = self.model.cpu()
        # super(DefaultPredictor, self).__init__()

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

    def batch_detect(self, original_images):
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
                original_images = original_images[..., ::-1]
            height, width = original_images.shape[1:3]
            inputs = []
            bs = original_images.shape[0]
            for original_image in original_images:
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width})
            predictions = self.model(inputs)
            return predictions

    def batch_get_feat(self, original_images, objects, objects_class):
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
                original_images = original_images[:, :, :, ::-1]
            height, width = original_images.shape[1:3]
            inputs = []
            detected_instances = []
            bs = original_images.shape[0]
            num_obj = objects.shape[1]
            for original_image, object, object_category in zip(original_images, objects, objects_class):
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width})
                aa = Instances(image.shape)
                object[:, 0] = object[:, 0] / width * image.shape[2]
                object[:, 1] = object[:, 1] / height * image.shape[1]
                object[:, 2] = object[:, 2] / width * image.shape[2]
                object[:, 3] = object[:, 3] / height * image.shape[1]
                boxes = Boxes(object)
                aa.set("pred_boxes", boxes)
                aa.set("pred_classes", object_category)
                detected_instances.append(aa)
            # predictions = self.model(inputs)[0]
            feats = self.model(inputs, detected_instances)
            return feats.reshape(bs * num_obj, -1)

def detector_setup_cfg(config, project_dir, yaml_dir, pkl_dir):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(project_dir, yaml_dir))
    cfg.merge_from_list(["MODEL.WEIGHTS", os.path.join(project_dir, pkl_dir), "INPUT.FORMAT", "RGB"])
    # Set score_threshold for builtin modelsdetector
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = config.detector_th
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.detector_th
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = config.detector_th
    cfg.freeze()
    return cfg

class Detector(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        # super(DetectorWrapper, self).__init__(env)
        args = SimpleNamespace(**cfg['ARGS'])
        self.cfg = cfg
        self.args = args
        detector_cfg = detector_setup_cfg(cfg, args.project_dir, "model/Detector/mask_rcnn_R_50_FPN_3x.yaml", "data/detector/model_final_f10217.pkl")

        self.metadata = MetadataCatalog.get(
            detector_cfg.DATASETS.TEST[0] if len(detector_cfg.DATASETS.TEST) else "__unused"
        )
        # self.cpu_device = torch.device("cpu")
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
        return objects, scores, classes

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

    def run_on_batch(self, images):
        """
        Args:
            image (np.ndarray): an image of shape (B, H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            objects, scores, classes: the output of the model.
        """
        predictions = self.predictor.batch_detect(images)
        B, H, W, C = images.shape
        objects_out = []
        scores_out = []
        classes_out = []
        for pred in predictions:
            objects = pred['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
            scores = pred['instances']._fields['scores'].cpu().detach().numpy()
            classes = pred['instances']._fields['pred_classes'].cpu().detach().numpy()
            if len(objects) > 0:
                objects[:, 0] = np.maximum(0, objects[:, 0])
                objects[:, 1] = np.maximum(0, objects[:, 1])
                objects[:, 2] = np.minimum(W-1, objects[:, 2])
                objects[:, 3] = np.minimum(H-1, objects[:, 3])
                score_idx = np.argsort(-scores)#[:100]
                objects_out.append(objects[score_idx])
                scores_out.append(scores[score_idx])
                classes_out.append(classes[score_idx])
            else:
                objects_out.append(objects)
                scores_out.append(scores)
                classes_out.append(classes)
        del predictions
        return objects_out, scores_out, classes_out



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
        # self.cpu_device = torch.device("cpu")
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
                # rois_inside = ((objects[:, 2] - objects[:, 0]) > 20) & (objects[:, 3] - objects[:, 1] > 20)
                # objects = objects[rois_inside]
                # classes = classes[rois_inside]
                # scores = scores[rois_inside]
        return objects, scores, classes

    def run_on_batch(self, images):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor.batch(images)
        objects = predictions['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
        scores = predictions['instances']._fields['scores'].cpu().detach().numpy()
        classes = predictions['instances']._fields['pred_classes'].cpu().detach().numpy()
        del predictions
        # image_shape = image.shape
        # objects[:, 0] = objects[:, 0] / image_shape[1]
        # objects[:, 2] = objects[:, 2] / image_shape[1]
        # objects[:, 1] = objects[:, 1] / image_shape[0]
        # objects[:, 3] = objects[:, 3] / image_shape[0]
        return objects, scores, classes

#
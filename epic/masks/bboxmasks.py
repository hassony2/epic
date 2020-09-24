import torch
from detectron2.data import transforms
from detectron2 import config as detcfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances, Boxes
from epic.masks import coco
from epic import boxutils


class MaskExtractor:
    def __init__(self):
        cfg = detcfg.get_cfg()
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        # sel_classes = ["frisbee", "baseball bat", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "hot dog", "pizza", "donut", "cake", "potted plant", "tv", "laptop", "mouse", "remote", "cell phone", "book", "clock", "vase", "scissors"]
        sel_classes = ["frisbee"]
        self.thing_idxs = [coco.class_names.index(cls) - 1 for cls in sel_classes]

    def preprocess_img(self, original_image, input_format="BGR"):
        if input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        aug = transforms.ResizeShortestEdge(
            [self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST],
            self.cfg.INPUT.MAX_SIZE_TEST,
        )
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).cuda()

        inputs = {"image": image, "height": height, "width": width}
        return inputs

    def masks_from_bboxes(
        self, im, boxes, pred_classes=None, class_idx=1, input_format="BGR"
    ):
        model = self.predictor.model

        # Initialize boxes
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.Tensor(boxes)
        boxes = Boxes(boxes)
        if pred_classes is None:
            pred_classes = class_idx * torch.ones(len(boxes)).long()
        else:
            if not isinstance(pred_classes, torch.Tensor):
                pred_classes = torch.Tensor(pred_classes)
            pred_classes = pred_classes.long()
        instances = Instances(
            image_size=(im.shape[0], im.shape[1]),
            pred_boxes=boxes,
            pred_classes=pred_classes,
        )

        # Preprocess image
        inp_im = self.preprocess_img(im, input_format=input_format)
        inf_out = model.inference([inp_im], [instances])

        # Extract masks
        instance = inf_out[0]["instances"]
        masks = instance.pred_masks
        boxes = instance.pred_boxes.tensor
        scores = instance.scores
        pred_classes = instance.pred_classes
        class_names = [coco.class_names[cls + 1] for cls in pred_classes]
        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "classes": class_names,
        }

    def masks_from_df(self, img, df, input_format="RGB", resize_factor=1):
        obj_df = df[df.det_type == "object"]
        if len(obj_df):
            boxes = []
            # For input to mask-rcnn
            pred_classes = []
            pred_boxes = []
            for _, row in obj_df.iterrows():
                box = boxutils.dfbox_to_norm(row, resize_factor=resize_factor)
                boxes.append(box)

                for cls in self.thing_idxs:
                    pred_boxes.append(box)
                    pred_classes.append(cls)
            res = self.masks_from_bboxes(
                img, pred_boxes, pred_classes=pred_classes, input_format=input_format
            )
            masks = res["masks"]
            masks = [mask.sum(0) for mask in masks.split(len(self.thing_idxs))]
        else:
            masks = []
            boxes = []
        return masks, boxes

    def img_inference(self, img, input_format="BGR", score_thresh=0.5):
        model = self.predictor.model

        # Preprocess image
        inp_im = self.preprocess_img(img, input_format=input_format)
        inf_out = model.inference([inp_im])

        # Extract masks
        masks = inf_out[0]["instances"].pred_masks
        boxes = inf_out[0]["instances"].pred_boxes.tensor
        scores = inf_out[0]["instances"].scores
        pred_classes = inf_out[0]["instances"].pred_classes
        class_names = [coco.class_names[cls + 1] for cls in pred_classes]
        keep_idxs = [idx for idx, score in enumerate(scores) if score > score_thresh]
        boxes = torch.stack([boxes[idx] for idx in keep_idxs])
        scores = torch.stack([scores[idx] for idx in keep_idxs])
        masks = torch.stack([masks[idx] for idx in keep_idxs])
        classes = [class_names[idx] for idx in keep_idxs]
        return {"masks": masks, "boxes": boxes, "scores": scores, "classes": classes}

from iou import calculate_iou
import torch


def non_max_suppression(boxes, threshold=0.5, iou_threshold=0.5, box_format="corners"):
    # boxes = [[class_no, probability, x1, y1, x2, y2]]
    bboxes = [box for box in boxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen = bboxes.pop(0)
        bboxes = [bbox for bbox in bboxes if chosen[0] != bbox[0] or calculate_iou(
            torch.tensor(chosen[2:]), torch.tensor(bbox[2:]), box_format) < iou_threshold]

        bboxes_after_nms.append(chosen)

    return bboxes_after_nms

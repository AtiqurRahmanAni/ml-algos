from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch import Tensor


def calculate_iou(box1, box2, box_format="midpoint"):
    if box_format == "midpoint":
        # [mid_x, mid_y, w, h]
        # top left
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        # bottom right
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2

        # top left
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        # bottom right
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2

    else:
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]

        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)

    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    iou = intersection / (box1_area + box2_area - intersection + 1e-6)
    return iou


def mean_average_precision(pred_boxes: list, true_boxs: list,  iou_threshold: 0.5, box_format="corners", num_classes=20):
    # pred_boxes = [[train_idx, class_pred, prob_score, x1, y1, x2, y2]]
    # true_boxes = [[train_idx, class_pred, prob_score, x1, y1, x2, y2]]

    avg_precision = []
    eps = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxs:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            # amount_bboxes = {0: torch.tensor([0, 0, 0]), 1: torch.tensor([0, 0, 0, 0])}
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP, FP = torch.zeros(len(detections)), torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            # taking ground truth bbox for a particular image
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = calculate_iou(torch.tensor(detection[3:]),
                                    torch.tensor(gt[3:]), box_format)

                # taking the best one
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + eps)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
        precisions = torch.cat(((torch.tensor([1])), precisions))
        recalls = torch.cat(((torch.tensor([0])), recalls))
        avg_precision.append(torch.trapz(precisions, recalls))

    return sum(avg_precision) / len(avg_precision)


def non_max_suppression(boxes, threshold=0.5, iou_threshold=0.5, box_format="corners"):
    # boxes = [[class_no, box_probability, x1, y1, x2, y2]]
    bboxes = [box for box in boxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen = bboxes.pop(0)
        bboxes = [bbox for bbox in bboxes if chosen[0] != bbox[0] or calculate_iou(
            torch.tensor(chosen[2:]), torch.tensor(bbox[2:]), box_format) < iou_threshold]

        bboxes_after_nms.append(chosen)

    return bboxes_after_nms


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.axis("off")
    plt.show()


def cellboxes_to_boxes(predictions, S=7):
    '''
    convert bounding boxes output from yolo 
    to bounding box relative to images
    for each image, the output is
    [predicted_class, best_confident, converted_bboxes] 
    '''

    batch_result = []
    for batch in predictions:
        all_bboxes = []
        # iterate every split S of an image
        for i in range(S):
            for j in range(S):
                prediction = batch[i, j].tolist()
                bbox1 = prediction[21:25]
                bbox2 = prediction[26:30]

                # taking the best bbox bases on box confident
                best_box = bbox1 if prediction[20] > prediction[25] else bbox2

                # convert the box position form w.r.t cell to the entire image
                # best_box = [x, y, w, h]
                x = (best_box[0] + j) / S
                y = (best_box[1] + i) / S
                w = best_box[2] / S
                h = best_box[3] / S

                predicted_class = np.argmax(prediction[:20], axis=0)
                best_box_confident = max(prediction[20], prediction[25])
                all_bboxes.append(
                    [predicted_class, best_box_confident, x, y, w, h])

        # applying non_max_suppression
        all_bboxes = non_max_suppression(
            all_bboxes, threshold=0.5, iou_threshold=0.5, box_format='midpoint')
        batch_result.append(all_bboxes)

    return batch_result

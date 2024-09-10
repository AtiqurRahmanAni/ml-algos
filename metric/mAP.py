from iou import calculate_iou
from collections import Counter
import torch


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

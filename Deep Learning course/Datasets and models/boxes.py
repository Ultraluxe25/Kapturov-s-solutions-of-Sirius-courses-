import torch


def get_bound_boxes(images, model, iou, iou_threshold=.5, conf_threshold=.4):

    predictions = model(images)

    class_1 = (predictions[..., -3] > conf_threshold).float().unsqueeze(3).expand(predictions.shape)
    class_2 = (predictions[..., -2] > conf_threshold).float().unsqueeze(3).expand(predictions.shape)
    class_3 = (predictions[..., -1] > conf_threshold).float().unsqueeze(3).expand(predictions.shape)

    true_boxes = []

    boxes_1 = (predictions * class_1)[..., :-3].flatten()
    boxes_1_check = []
    for i in range(0, len(boxes_1), 5):
        if torch.sum(boxes_1[i:i+5]) != 0:
            boxes_1_check.append(boxes_1[i:i+5].data)

    bboxes = [box for box in boxes_1_check if box[4] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda box: box[4], reverse=True)
    bboxes_after_nms = []
    i = 0
    while i < len(bboxes) - 1:
        iou_res = iou(
            bboxes[i], bboxes[i + 1]
        )
        bboxes_after_nms.append(bboxes[i])
        if iou_res > iou_threshold:
            i += 2
        else:
            i += 1

    true_boxes.extend(bboxes_after_nms)

    boxes_2 = (predictions * class_2)[..., :-3].flatten()
    boxes_2_check = []
    for i in range(0, len(boxes_2), 5):
        if torch.sum(boxes_2[i:i+5]) != 0:
            boxes_2_check.append(boxes_2[i:i+5].data)

    bboxes = [box for box in boxes_2_check if box[4] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda box: box[4], reverse=True)
    bboxes_after_nms = []
    i = 0
    while i < len(bboxes) - 1:
        iou_res = iou(
            bboxes[i], bboxes[i + 1]
        )
        bboxes_after_nms.append(bboxes[i])
        if iou_res > iou_threshold:
            i += 2
        else:
            i += 1

    true_boxes.extend(bboxes_after_nms)

    boxes_3 = (predictions * class_3)[..., :-3].flatten()
    boxes_3_check = []
    for i in range(0, len(boxes_3), 5):
        if torch.sum(boxes_3[i:i+5]) != 0:
            boxes_3_check.append(boxes_3[i:i+5].data)

    bboxes = [box for box in boxes_3_check if box[4] > conf_threshold]
    bboxes = sorted(bboxes, key=lambda box: box[4], reverse=True)
    bboxes_after_nms = []
    i = 0
    while i < len(bboxes) - 1:
        iou_res = iou(
            bboxes[i], bboxes[i + 1]
        )
        bboxes_after_nms.append(bboxes[i])
        if iou_res > iou_threshold:
            i += 2
        else:
            i += 1

    true_boxes.extend(bboxes_after_nms)

    return true_boxes
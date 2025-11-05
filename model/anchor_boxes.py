import torch
import math

# use gpu if available for faster computation (ran on cuda 13.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def generate_anchor_boxes(
    feature_maps: list[torch.Tensor], 
    aspect_ratios: list[float] = [1.0, 2.0, 3.0, 0.5, 0.33], 
) -> torch.Tensor:
    '''
    Generate the default anchor boxes for object detection, as described in https://arxiv.org/pdf/1512.02325.
    Default values are set to those used in the paper.
    
    Args:
        **feature_map**: list of input tensors of shape (batch_size, channels, height, width); feature maps
        generated from all convolutional layers of the model
        **aspect_ratios**: list of aspect ratios (width / height) for the boxes
        **num_scales**: number of different scales for the boxes
        
    Returns:
        **default_boxes**: tensor of shape (batch_size, num_boxes, 4) representing the default anchor boxes in
        (xmin, ymin, xmax, ymax) format, normalized between 0 and 1
    '''
    # evenly spaced scales between 0.2 and 0.9, as described in the paper
    scales = torch.linspace(0.2, 0.9, len(feature_maps), device=device)
    ratios = torch.tensor(aspect_ratios, device=device)
    boxes = []
    for k in range(len(feature_maps)):
        # s'_k = sqrt(s_k * s_(k+1)) as described in the paper, only for scale=1
        s_prime = math.sqrt(scales[k] * scales[k + 1]) if k + 1 < len(feature_maps) else scales[k]
        pairs = [(s_prime, s_prime)]
        # generate width-height pairs for each aspect ratio
        for r in ratios:
            w = scales[k] * math.sqrt(r)
            h = scales[k] / math.sqrt(r)
            pairs.append((w, h))
        
        input_h, input_w = feature_maps[k].shape[-2:]
        
        # generate normalized center coordinates for each box, for scale invariance since coords are based
        # on proportion of image size, not absolute pixel values
        center_h = (torch.arange(input_h, device=device) + 0.5) / input_h
        center_w = (torch.arange(input_w, device=device) + 0.5) / input_w
        # turns into 2 grids of shape (input_h, input_w), broadcasting columns and rows respectively
        # shift_y will have each row be the same, shift_x will have each column be the same
        # this is because each row corresponds to a different y center, each column to a different x center
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        # reshape to (input_h * input_w, 1), the shifts are the centers of each box, reshape(-1) flattens
        shift_y = shift_y.reshape(-1)
        shift_x = shift_x.reshape(-1)
        # shape (num_boxes * num_pixels, 2), where each row is (shift_x, shift_y) for a box
        shifts = torch.stack((shift_x, shift_y), dim=1)
        shifts = shifts.repeat_interleave(len(pairs), dim=0)
        # turns pairs into the same shape as shifts, then concatenates to get 
        # (shift_x, shift_y, width, height)
        pairs = torch.as_tensor(pairs, device=device)
        pairs = pairs.repeat(input_h * input_w, 1)
        box = torch.cat((shifts, pairs), dim=1)
        
        # convert from (center_x, center_y, width, height) to (xmin, ymin, xmax, ymax)
        center_x, center_y, width, height = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        xmin = center_x - width / 2
        ymin = center_y - height / 2
        xmax = center_x + width / 2
        ymax = center_y + height / 2
        box = torch.stack([xmin, ymin, xmax, ymax], dim=1)
        
        boxes.append(box)
    
    all_boxes = torch.cat(boxes, dim=0)
    
    batch_size = feature_maps[0].shape[0]

    # unsqueeze adds a dimension in front, repeat boxes for each batch and reshape to 
    # (batch_size, num_boxes, 4), so one set of boxes per image in the batch
    default_boxes = all_boxes.unsqueeze(0).expand(batch_size, -1, -1)
    
    return default_boxes

def iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    '''
    Computes the intersection over union between two sets of boxes. The IOU is defined as the area of
    the intersection divided by the area of the union of two boxes. This is also called the Jaccard index.
    
    Args:
        **box_a**: tensor of shape (num_boxes_a, 4) representing boxes in (xmin, ymin, xmax, ymax) format
        **box_b**: tensor of shape (num_boxes_b, 4) representing boxes in (xmin, ymin, xmax, ymax) format
    
    Returns:
        **iou**: tensor of shape (num_boxes_a, num_boxes_b) representing the pairwise IOU between each box 
        in boxes_a and boxes_b
    '''
    areas_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    areas_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    # max gets the rightmost left x and bottommost top y
    # None in the indexing adds a new dimension (num_boxes_a,) -> (num_boxes_a, 1)
    # broadcasts from (num_boxes_a, 1) and (num_boxes_b,) to (num_boxes_a, num_boxes_b)
    # pairwise matrices of left x and top y coordinates of the intersection boxes
    x_left = torch.max(boxes_a[:, None, 0], boxes_b[:, 0])
    y_top = torch.max(boxes_a[:, None, 1], boxes_b[:, 1])
    # same as above but for leftmost right x and topmost bottom y
    x_right = torch.min(boxes_a[:, None, 2], boxes_b[:, 2])
    y_bottom = torch.min(boxes_a[:, None, 3], boxes_b[:, 3])

    # values are clamped since if x_right < x_left or y_bottom < y_top, there is no intersection
    intersection_areas = torch.clamp(x_right - x_left, min=0) * torch.clamp(y_bottom - y_top, min=0)
    # similar as before, pairwise matrix of union areas between each pair of boxes
    union_areas = areas_a[:, None] + areas_b - intersection_areas
    iou = intersection_areas / union_areas

    return iou

def assign_bounding_boxes(
    ground_truths: torch.Tensor, 
    anchor_boxes: torch.Tensor, 
    iou_threshold: float = 0.5
) -> torch.Tensor:
    '''
    Calculates the jaccards (IOU) between each anchor box and ground truth box, assigns each anchor to 
    the ground truth box with the highest IOU if it is above the threshold, then assigns remaining unassigned ground 
    truth boxes to the anchor with the highest IOU. Details can be seen here:
    https://d2l.ai/chapter_computer-vision/anchor.html#:~:text=to%20anchor%0Aboxes.-,14.4.3.1,-.%20Assigning%20Ground-Truth
    
    Args:
        **ground_truths**: tensor of shape (num_ground_truths, 4) representing ground truth bounding boxes in 
        (xmin, ymin, xmax, ymax) format
        **anchor_boxes**: tensor of shape (num_anchors, 4) representing anchor boxes in 
        (xmin, ymin, xmax, ymax) format
        **iou_threshold**: minimum Jaccard index required to assign an anchor box to a ground truth bounding box
    
    Returns:
        **anchor_bounding_boxes**: tensor of shape (num_anchors,) representing the index of the assigned 
        ground truth box for each anchor box; -1 if no ground truth box is assigned
    '''
    n_anchors, n_ground_truths = anchor_boxes.shape[0], ground_truths.shape[0]
    jaccards = iou(anchor_boxes, ground_truths)
    
    anchor_bounding_boxes = torch.full((n_anchors,), -1, dtype=torch.int64, device=device)
    
    # for each anchor box, get the ground truth box with the highest jaccard index, indices are column indices
    # i.e. for each row get the value and index of the max value in that row
    # multiple anchors can have the same ground truth box assigned here
    max_jaccards, indices = torch.max(jaccards, dim=1)
    # gets the row indices for which the max jaccard is above the threshold
    anchor_indices = torch.nonzero(max_jaccards >= iou_threshold).reshape(-1)
    # gets the column indices for which the max jaccard is above the threshold
    box_indices = indices[max_jaccards >= iou_threshold]
    anchor_bounding_boxes[anchor_indices] = box_indices
    # now, all anchors with IOU above the threshold have been assigned a ground truth box
    
    col_discard = torch.full((n_anchors,), -1, dtype=torch.int64, device=device)
    row_discard = torch.full((n_ground_truths,), -1, dtype=torch.int64, device=device)
    
    # for each ground truth box, find the anchor box with the highest jaccard index and assign regardless
    # of whether it is above the threshold
    for i in range(n_ground_truths):
        # without dim argument, this flattens row by row and returns the index of the overall max
        max_index = torch.argmax(jaccards)
        # is correct since the first n_ground_truths values correspond to anchor 0, 
        # next n_ground_truths to anchor 1, etc.
        box_index = max_index % n_ground_truths
        anchor_index = max_index // n_ground_truths
        
        anchor_bounding_boxes[anchor_index] = box_index
        jaccards[:, box_index] = col_discard
        jaccards[anchor_index, :] = row_discard
    # this ensures that not only are all anchors with IOU above the threshold assigned a ground truth box,
    # but also that each ground truth box is assigned to at least one anchor box
    # some anchors may remain unassigned if their IOU is below the threshold and they are not the 
    # best match for any ground truth box
    
    return anchor_bounding_boxes

def encode_offsets(
    anchor_boxes: torch.Tensor, 
    ground_truths: torch.Tensor, 
    anchor_bounding_boxes: torch.Tensor, 
    variances: list[float] = [0.1, 0.2]
) -> torch.Tensor:
    '''
    Encode the offsets between the anchor boxes and the assigned ground truth boxes using the variances. 
    Method is described in https://arxiv.org/pdf/1311.2524 Appendix C. However, in SSDs, the "proposals" 
    are the anchor boxes and variances are applied to the encoded offsets for better training stability.
    
    Args:
        **anchor_boxes**: tensor of shape (num_anchors, 4) representing anchor boxes in 
        (xmin, ymin, xmax, ymax) format
        **ground_truths**: tensor of shape (num_ground_truths, 4) representing ground truth bounding boxes in 
        (xmin, ymin, xmax, ymax) format
        **anchor_bounding_boxes**: tensor of shape (num_anchors,) representing the index of the assigned 
        ground truth box for each anchor box; -1 if no ground truth box is assigned
        **variances**: list of two float values representing the variances used for encoding
    
    Returns:
        **encoded_boxes**: tensor of shape (num_anchors, 4) representing the encoded boxes (offsets between 
        anchor boxes and assigned ground truth boxes)
    '''
    encoded_boxes = torch.zeros((anchor_boxes.shape[0], 4), device=device)
    
    # filter out the unassigned anchors
    mask = anchor_bounding_boxes >= 0
    anchor_boxes = anchor_boxes[mask]
    assigned_indices = anchor_bounding_boxes[mask]
    ground_truths = ground_truths[assigned_indices]
    
    # don't waste time computing if no anchors are assigned
    if mask.sum() == 0:
        return encoded_boxes
    
    # convert to cx, cy, w, h format
    anchor_cx = (anchor_boxes[:, 0] + anchor_boxes[:, 2]) / 2
    anchor_cy = (anchor_boxes[:, 1] + anchor_boxes[:, 3]) / 2
    anchor_w = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_h = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    
    ground_truth_cx = (ground_truths[:, 0] + ground_truths[:, 2]) / 2
    ground_truth_cy = (ground_truths[:, 1] + ground_truths[:, 3]) / 2
    ground_truth_w = ground_truths[:, 2] - ground_truths[:, 0]
    ground_truth_h = ground_truths[:, 3] - ground_truths[:, 1]

    # normalized for size invariance 
    dx = (ground_truth_cx - anchor_cx) / anchor_w
    dy = (ground_truth_cy - anchor_cy) / anchor_h
    # the log is taken so that, for example, multiplying and dividing by 2 are symmetric operations
    dw = torch.log(ground_truth_w / anchor_w)
    dh = torch.log(ground_truth_h / anchor_h)
    
    # variances are larger for w and h because these values tend to vary more, and we want the encoded values
    # for w and h to be scaled down more to prevent instability during training
    var_cx, var_cy = variances[0], variances[0]
    var_w, var_h = variances[1], variances[1]
    
    offsets_cx = dx / var_cx
    offsets_cy = dy / var_cy
    offsets_w = dw / var_w
    offsets_h = dh / var_h

    # Apply the variances
    encoded_boxes[mask] = torch.stack((offsets_cx, offsets_cy, offsets_w, offsets_h), dim=1)

    return encoded_boxes

# TODO: implement decoding, NMS, post-processing (do after model.py is done)
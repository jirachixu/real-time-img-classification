import torch
import math

# use gpu if available for faster computation (ran on cuda 13.0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def generate_boxes(
    feature_map: list[torch.Tensor], 
    aspect_ratios: list[float] = [1.0, 2.0, 3.0, 0.5, 0.33], 
    num_scales: int = 5
) -> torch.Tensor:
    '''
    Generate the default anchor boxes for object detection, as described in https://arxiv.org/pdf/1512.02325.
    Default values are set to those used in the paper.
    
    Args:
        **feature_map**: list of input tensors of shape (batch_size, channels, height, width); feature maps
        generated from all convolutional layers of the model
        **aspect_ratios**: list of aspect ratios (width / height) for the boxes
        **num_scales**: number of different scales for the boxes
    '''
    # evenly spaced scales between 0.2 and 0.9, as described in the paper
    scales = torch.linspace(0.2, 0.9, num_scales, device=device)
    ratios = torch.tensor(aspect_ratios, device=device)
    boxes = []
    for k in range(len(feature_map)):
        # s'_k = sqrt(s_k * s_(k+1)) as described in the paper, only for scale=1
        s_prime = math.sqrt(scales[k] * scales[k + 1]) if k + 1 < num_scales else scales[k]
        pairs = [(s_prime, s_prime)]
        # generate width-height pairs for each aspect ratio
        for r in ratios:
            w = scales[k] * math.sqrt(r)
            h = scales[k] / math.sqrt(r)
            pairs.append((w, h))
        
        input_h, input_w = feature_map[k].shape[-2:]
        
        # generate normalized center coordinates for each box
        center_h = (torch.arange(input_h, device=device) + 0.5) / input_h
        center_w = (torch.arange(input_w, device=device) + 0.5) / input_w
        # turns into 2 grids of shape (input_h, input_w), broadcasting columns and rows respectively
        # shift_y will have each row be the same, shift_x will have each column be the same
        # this is because each row corresponds to a different y center, each column to a different x center
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        # reshape to (input_h * input_w, 1), the shifts are the centers of each box
        shift_y = shift_y.reshape(-1)
        shift_x = shift_x.reshape(-1)
        # shape (num_boxes * num_pixels, 2), where each row is (shift_x, shift_y) for a box
        shifts = torch.stack((shift_x, shift_y) * len(pairs), dim=1).reshape(-1, 2)
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
    
    batch_size = feature_map[0].shape[0]
    
    # repeat boxes for each batch and reshape to (batch_size, num_boxes, 4)
    default_boxes = all_boxes.unsqueeze(0).expand(batch_size, -1, -1)
    
    return default_boxes

print(generate_boxes([torch.randn(2, 3, 300, 300).to(device), torch.randn(2, 3, 150, 150).to(device)]).shape)

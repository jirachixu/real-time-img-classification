import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SSDModel(nn.Module):
    '''
    SSD model using ResNet-50 as backbone. Architecture is described in the SSD paper.
        
    Args:
        n_classes (int): Number of object classes (including background).
    '''
    def __init__(self, n_classes: int = 80) -> None:
        super().__init__()
        # +1 for background class
        self.n_classes = n_classes + 1
        self.n_anchors = 6 # number of anchors per feature map location
        # uses ResNet-50 to generate feature maps, we need to use the pretrained weights otherwise the 
        # feature maps will be meaningless
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.to(device)
        # layer1 contains non-feature-extracting layers
        self.layer1 = nn.Sequential(
            # remove avgpool and fully connected layers and last 3 conv blocks, since we need their outputs
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        # these layers extract the feature maps we will use for detection
        # dimensions specified by the input size of 300 x 300, as in the paper
        # https://cs231n.github.io/convolutional-networks/#conv:~:text=architectures%20section%20below.-,Convolution%20Demo,-.%20Below%20is%20a
        # demo of convolution operation on matrix
        self.layer2 = self.backbone.layer2 # 38 x 38
        self.layer3 = self.backbone.layer3 # 19 x 19
        self.layer4 = self.backbone.layer4 # 10 x 10
        self.extra1 = self._make_extra_layers(2048, 256, 512) # 5 x 5
        self.extra2 = self._make_extra_layers(512, 128, 256) # 3 x 3
        self.extra3 = self._make_extra_layers(256, 128, 256, padding=0, stride=1) # 1 x 1
        
        # detection heads that predict offsets for each feature map
        self.offset_convs = nn.ModuleList([
            # we multiply by 4 because we need to predict 4 offsets per anchor box
            # ∆cx, ∆cy, ∆w, ∆h
            # outputs will have shape (batch_size, n_anchors * 4, h, w)
            nn.Conv2d(512, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(2048, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * 4, kernel_size=3, padding=1),
        ])
        
        self.classification_convs = nn.ModuleList([
            # we multiply by the number of classes because we need to predict a class score for 
            # each anchor box per location
            # outputs will have shape (batch_size, n_anchors * n_classes, h, w)
            nn.Conv2d(512, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(2048, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, self.n_anchors * self.n_classes, kernel_size=3, padding=1),
        ])
        
    def _make_extra_layers(self, 
            in_channels: int, 
            bottleneck_channels: int, 
            out_channels: int,
            padding: int = 1, 
            stride: int = 2
        ) -> nn.Sequential:
        '''
        Helper to create extra conv blocks, which are used to create the smallest feature maps. Based on the 
        bottleneck blocks in ResNet, which significantly reduce parameter count. The original SSD 
        architecture in the paper uses 2 conv layers instead, but using 3 here further decreases parameter
        count.
        
        Args:
            in_channels (int): Number of input channels.
            bottleneck_channels (int): Number of channels in the bottleneck layer.
            out_channels (int): Number of output channels.
            padding (int, optional): Padding for the 3x3 convolution.
            stride (int, optional): Stride for the 3x3 convolution.
            
        Returns:
            nn.Sequential: Extra bottleneck convolutional block.
        '''
        return nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1),
        )
        
    # TODO: implement loss function and prediction function
    def forward(self, img: torch.Tensor, softmax: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass through the model. Turns the input image into feature maps, then applies the 
        corresponding detection heads for each feature map to get the final predictions.
        
        Args:
            img (torch.Tensor): Input image tensor of shape (batch_size, 3, H, W).
            softmax (bool, optional): Whether to apply softmax to class scores. Defaults to False.
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple containing offsets tensor and class scores tensor.
        '''
        feat = self.layer1(img)
        feat_1 = self.layer2(feat)
        feat_2 = self.layer3(feat_1)
        feat_3 = self.layer4(feat_2)
        feat_4 = self.extra1(feat_3)
        feat_5 = self.extra2(feat_4)
        feat_6 = self.extra3(feat_5)
        feature_maps = [feat_1, feat_2, feat_3, feat_4, feat_5, feat_6]
        
        all_offsets = []
        all_class_scores = []
        
        for i, feature_map in enumerate(feature_maps):
            offsets = self.offset_convs[i](feature_map)
            class_scores = self.classification_convs[i](feature_map)
            # we have to reshape here to allow for concatenation
            # torch.Tensor.contiguous() returns a contiguous tensor containing the same data 
            # as self tensor, which is more efficient than a non-contiguous tensor
            # also allows us to use the more efficient view() function instead of reshape()
            offsets = offsets.permute(0, 2, 3, 1).contiguous()
            class_scores = class_scores.permute(0, 2, 3, 1).contiguous()
            # (batch_size, n_anchors * h * w, 4)
            # flattens dimensions so that the middle dimension becomes each of the anchors' 
            # offsets at (0, 0), (0, 1), ..., (h, w), and then the 4 at the end takes every 
            # 4 consecutive values and groups them (∆cx, ∆cy, ∆w, ∆h)
            offsets = offsets.view(offsets.shape[0], -1, 4) 
            # (batch_size, n_anchors * h * w, n_classes)
            class_scores = class_scores.view(class_scores.shape[0], -1, self.n_classes)
            
            all_offsets.append(offsets)
            all_class_scores.append(class_scores)
        
        all_offsets = torch.cat(all_offsets, dim=1)
        all_class_scores = torch.cat(all_class_scores, dim=1)
        
        if softmax:
            all_class_scores = F.softmax(all_class_scores, dim=-1)
        
        return all_offsets, all_class_scores

# test dimensionality of outputs
model = SSDModel()
model.to(device)

with torch.no_grad():
    x = torch.randn(1, 3, 300, 300, device=device)
    
    forwarded = model.forward(x)
    print(forwarded[0].shape, forwarded[1].shape)
    
    x = model.layer1(x)
    
    x = model.layer2(x)
    print(f"Layer2 output: {x.shape}")
    x = model.layer3(x)
    print(f"Layer3 output: {x.shape}")
    x = model.layer4(x)
    print(f"Layer4 output: {x.shape}")
    x = model.extra1(x)
    print(f"Extra1 output: {x.shape}")
    x = model.extra2(x)
    print(f"Extra2 output: {x.shape}")
    x = model.extra3(x)
    print(f"Extra3 output: {x.shape}")

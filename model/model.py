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
        # uses ResNet-50 TODO write comment about why
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
        
        # TODO: implement forward pass, loss function, and prediction function

# test dimensionality of outputs
model = SSDModel()
model.to(device)

with torch.no_grad():
    x = torch.randn(1, 3, 300, 300, device=device)
    
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

# models.py
import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNetWithDropout(nn.Module):
    def __init__(self, base_name: str, num_classes: int, dropout_p: float = 0.0):
        super().__init__()

        if base_name == "resnet18":
            base = tvm.resnet18(weights=None)
        elif base_name == "resnet34":
            base = tvm.resnet34(weights=None)
        else:
            raise ValueError(f"Unsupported ResNet: {base_name}")

        # Replace the final fully connected layer with Dropout + Linear
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        self.base = base

    def forward(self, x):
        return self.base(x)


class MobileNetV3WithDropout(nn.Module):
    def __init__(self, variant: str, num_classes: int, dropout_p: float = 0.0):
        """
        Wrap torchvision MobileNetV3 and only modify the classifier.
        We DO NOT manually flatten or add our own fc; we let torchvision
        handle pooling and feature dimensions to avoid shape mismatches.
        """
        super().__init__()

        if variant == "small":
            base = tvm.mobilenet_v3_small(weights=None)
        elif variant == "large":
            base = tvm.mobilenet_v3_large(weights=None)
        else:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")

        # base.classifier is typically:
        # [0] Linear(last_channel -> hidden)
        # [1] Hardswish
        # [2] Dropout
        # [3] Linear(hidden -> num_classes)
        # We only swap the last Linear to match num_classes.
        last_linear = base.classifier[-1]
        in_features = last_linear.in_features
        base.classifier[-1] = nn.Linear(in_features, num_classes)

        # Set dropout probability for any Dropout inside classifier
        for m in base.classifier.modules():
            if isinstance(m, nn.Dropout):
                m.p = dropout_p

        self.base = base

    def forward(self, x):
        return self.base(x)


def create_model(model_name: str, num_classes: int, dropout_p: float = 0.0) -> nn.Module:
    """
    Factory function used by both centralized and federated training.
    """
    if model_name in ["resnet18", "resnet34"]:
        return ResNetWithDropout(model_name, num_classes, dropout_p=dropout_p)
    elif model_name == "mobilenetv3_small":
        return MobileNetV3WithDropout("small", num_classes, dropout_p=dropout_p)
    elif model_name == "mobilenetv3_large":
        return MobileNetV3WithDropout("large", num_classes, dropout_p=dropout_p)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def set_dropout_p(model: nn.Module, p: float):
    """
    Change dropout probability for all nn.Dropout layers in the model.
    This is what your federated code calls to set train-time dropout.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p

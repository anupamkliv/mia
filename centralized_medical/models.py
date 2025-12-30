from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import models


class MonteCarloDropout(nn.Dropout):
    """
    Dropout layer that can be kept active during inference by setting module.train()
    while still running under torch.no_grad().
    """
    def forward(self, input):
        return nn.functional.dropout(input, self.p, training=True, inplace=False)


def collect_mc_dropout_layers(model: nn.Module) -> List[MonteCarloDropout]:
    mc_layers = []
    for m in model.modules():
        if isinstance(m, MonteCarloDropout):
            mc_layers.append(m)
    return mc_layers


def set_mc_dropout_p(mc_layers: List[MonteCarloDropout], p: float) -> None:
    for m in mc_layers:
        m.p = p


def build_resnet_with_mc_dropout(arch: str, num_classes: int, p: float) -> nn.Module:
    if arch == "resnet18":
        base = models.resnet18(weights=None)
    elif arch == "resnet34":
        base = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported resnet arch: {arch}")

    in_features = base.fc.in_features
    base.fc = nn.Identity()

    class ResNetWithDropout(nn.Module):
        def __init__(self, backbone, in_features, num_classes, p):
            super().__init__()
            self.backbone = backbone
            self.dropout = MonteCarloDropout(p=p)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.backbone(x)
            feats = self.dropout(feats)
            logits = self.fc(feats)
            return logits

    return ResNetWithDropout(base, in_features, num_classes, p)


def build_mobilenetv3_with_mc_dropout(arch: str, num_classes: int, p: float) -> nn.Module:
    if arch == "mobilenetv3_small":
        base = models.mobilenet_v3_small(weights=None)
    elif arch == "mobilenetv3_large":
        base = models.mobilenet_v3_large(weights=None)
    else:
        raise ValueError(f"Unsupported mobilenetv3 arch: {arch}")

    in_features = base.classifier[-1].in_features
    base.classifier[-1] = nn.Identity()

    class MobileNetWithDropout(nn.Module):
        def __init__(self, backbone, in_features, num_classes, p):
            super().__init__()
            self.backbone = backbone
            self.dropout = MonteCarloDropout(p=p)
            self.fc = nn.Linear(in_features, num_classes)

        def forward(self, x):
            feats = self.backbone(x)
            feats = self.dropout(feats)
            logits = self.fc(feats)
            return logits

    return MobileNetWithDropout(base, in_features, num_classes, p)


def get_model(model_name: str, num_classes: int, dropout_p: float) -> Tuple[nn.Module, List[MonteCarloDropout]]:
    model_name = model_name.lower()

    if model_name in {"resnet18", "resnet34"}:
        model = build_resnet_with_mc_dropout(model_name, num_classes, dropout_p)
    elif model_name in {"mobilenetv3_small", "mobilenetv3_large"}:
        model = build_mobilenetv3_with_mc_dropout(model_name, num_classes, dropout_p)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    mc_layers = collect_mc_dropout_layers(model)
    return model, mc_layers

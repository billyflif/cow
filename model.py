import os
import json
import torch
import torch.nn as nn
import timm
from pathlib import Path
import logging
import torchvision.models as models
import numpy as np
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """难样本挖掘三元组损失 - 提升特征判别能力"""

    def __init__(self, margin=0.3):
        """
        参数:
            margin: 正负样本之间的边界距离
        """
        super().__init__()
        self.margin = margin
        self.eps = 1e-8

    def forward(self, embeddings, labels):
        """
        计算难样本挖掘三元组损失

        参数:
            embeddings: 特征嵌入张量, shape=(batch_size, embedding_size)
            labels: 样本标签, shape=(batch_size,)

        返回:
            三元组损失值
        """
        # 计算特征之间的欧氏距离矩阵
        dist_mat = self.euclidean_distance(embeddings)

        # 获取正负样本掩码
        mask_positive, mask_negative = self.get_masks(labels)

        # 计算最难正样本距离（同ID最大距离）
        positive_dist = (dist_mat * mask_positive).max(dim=1)[0]

        # 计算最难负样本距离（不同ID最小距离）
        negative_dist = (dist_mat + mask_negative * 1e9).min(dim=1)[0]

        # 计算三元组损失
        losses = F.relu(positive_dist - negative_dist + self.margin)
        return losses.mean()

    def euclidean_distance(self, x):
        """计算所有样本之间的欧氏距离矩阵"""
        square = torch.sum(x * x, dim=1)
        dist = torch.unsqueeze(square, 0) - 2 * torch.mm(x, x.t()) + torch.unsqueeze(square, 0).t()
        dist = F.relu(dist)  # 避免负距离
        return torch.sqrt(dist + self.eps)

    def get_masks(self, labels):
        """
        生成正负样本掩码

        返回:
            mask_positive: 正样本掩码 (同ID)
            mask_negative: 负样本掩码 (不同ID)
        """
        n = labels.size(0)
        # 创建标签比较矩阵
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

        # 正样本掩码（排除自身）
        mask_positive = labels_eq.float()
        mask_positive.fill_diagonal_(0)  # 排除自身

        # 负样本掩码
        mask_negative = (~labels_eq).float()
        mask_negative.fill_diagonal_(0)  # 排除自身

        return mask_positive, mask_negative

class CosFaceLoss(nn.Module):
    """CosFace损失函数实现"""

    def __init__(self, num_classes, embedding_size=512, margin=0.35, scale=64.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale
        
        # 权重矩阵初始化
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # 数值稳定性
        self.eps = 1e-7

    def forward(self, embeddings, labels):
        # L2归一化
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # 计算余弦相似度
        cos_theta = nn.functional.linear(embeddings, weight_norm)
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)
        
        # 为目标类别添加margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # CosFace的核心：直接在余弦值上减去margin
        margin_cos_theta = cos_theta - one_hot * self.margin
        
        # 应用scale因子
        scaled_cos_theta = margin_cos_theta * self.scale
        
        # 使用标签平滑以提高稳定性
        return nn.CrossEntropyLoss(label_smoothing=0.1)(scaled_cos_theta, labels)

class ArcFaceLoss(nn.Module):
    """改进的ArcFace损失函数 - 增强数值稳定性"""

    def __init__(self, num_classes, embedding_size=512, margin=0.5, scale=64.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.margin = margin
        self.scale = scale

        # 权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.th = np.cos(np.pi - margin)
        self.mm = self.sin_m * margin

        # 添加epsilon以提高数值稳定性
        self.eps = 1e-7

    def forward(self, embeddings, labels):
        # L2归一化
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)

        # 计算余弦相似度
        cos_theta = nn.functional.linear(embeddings, weight_norm)
        # 限制范围以防止数值不稳定
        cos_theta = cos_theta.clamp(-1 + self.eps, 1 - self.eps)

        # 计算角度
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2) + self.eps)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # 条件处理
        cond_v = cos_theta - self.th
        cond_mask = cond_v <= 0
        cos_theta_m[cond_mask] = (cos_theta - self.mm)[cond_mask]

        # 计算最终的logits
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        output *= self.scale

        # 使用标签平滑以提高稳定性
        return nn.CrossEntropyLoss(label_smoothing=0.1)(output, labels)


class LightweightCowModel(nn.Module):
    """轻量级备用模型（ResNet50基础）"""

    def __init__(self, num_classes=None, embedding_size=768):
        super().__init__()
        self.actual_model_name = 'resnet50'

        # 使用预训练的ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 移除最后的分类层

        # 添加投影头
        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, embedding_size),
            nn.BatchNorm1d(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, embedding_size)
        )

        self.num_classes = num_classes

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)
        return embeddings


class CowReIDModel(nn.Module):
    """牛重识别主模型 - 支持本地MegaDescriptor加载"""

    def __init__(self, model_name='MegaDescriptor-S-224', num_classes=None,
                 embedding_size=768, use_lightweight=False):
        super().__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.use_lightweight = use_lightweight
        self.logger = logging.getLogger(__name__)
        self.classifier = nn.Linear(embedding_size, num_classes) if num_classes else None

        if use_lightweight:
            self.logger.info("使用轻量级ResNet50模型")
            self._init_lightweight_model(num_classes, embedding_size)
            return

        try:
            self.logger.info(f"模型初始化")
            self.backbone, model_config = self.load_local_megadescriptor_model()
            self.actual_model_name = model_config['architecture']
            self._setup_embedding_layer()
            self.num_classes = num_classes
            self.logger.info(f"成功初始化模型")

            # self.logger.info(f"尝试加载本地模型: {model_name}")
            # self.backbone, model_config = self.load_local_megadescriptor_model()
            # self.actual_model_name = model_config['architecture']
            # self._setup_embedding_layer()
            # self.num_classes = num_classes
            # self.logger.info(f"成功加载模型: {self.actual_model_name}")

            # self.logger.info(f"从检查点加载模型")
            # self.backbone = self.load_local_checkpoint()
            # self.actual_model_name = "swin_small_patch4_window7_224"
            # self._setup_embedding_layer()
            # self.logger.info(f"成功加载模型")
        except Exception as e:
            self.logger.error(f"MegaDescriptor加载失败: {e}")
            self.logger.info("切换到轻量级模型")
            self._init_lightweight_model(num_classes, embedding_size)

    def load_local_megadescriptor_model(self):
        """直接从本地文件加载MegaDescriptor-S-224模型"""
        # model_dir = Path("../model_cache/models--BVRA--MegaDescriptor-S-224/snapshots/main")
        #
        # # 读取配置
        # config_path = model_dir / "config.json"
        # with open(config_path, 'r') as f:
        #     config = json.load(f)
        #
        # # 根据配置创建模型
        # model_name = config['architecture']
        config = {'architecture' : "swin_small_patch4_window7_224"}
        model_name = config['architecture']
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # 移除分类头
            global_pool='avg'
        )
        # # 尝试加载权重
        # weights_path = model_dir / "pytorch_model.bin"
        # if weights_path.exists():
        #     state_dict = torch.load(weights_path, map_location='cpu')
        #     model.load_state_dict(state_dict, strict=False)
        #
        # else:
        #     safetensors_path = model_dir / "model.safetensors"
        #     if safetensors_path.exists():
        #         from safetensors.torch import load_file
        #         state_dict = load_file(str(safetensors_path))
        #         model.load_state_dict(state_dict, strict=False)
        #     else:
        #         raise FileNotFoundError("找不到模型权重文件")
        return model, config

    def load_local_checkpoint(self):
        """直接从本地文件加载MegaDescriptor-S-224模型"""
        if self.model_dir:
            model_dir = Path(self.model_dir)
        else:
            model_dir = Path("E:\\COW\\Cow-Re-ID\\0722\\checkpoints\\model_20250805_172950\\best_model.pth")
        # 根据配置创建模型
        model_name = 'swin_small_patch4_window7_224'
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # 移除分类头
            global_pool='avg'
        )
        # 尝试加载权重
        weights_path = model_dir
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location='cpu')
            # print(state_dict['model_state_dict'].key())
            model.load_state_dict(state_dict['model_state_dict'], strict=False)
        else:
            safetensors_path = model_dir / "model.safetensors"
            if safetensors_path.exists():
                from safetensors.torch import load_file
                state_dict = load_file(str(safetensors_path))
                model.load_state_dict(state_dict, strict=False)
            else:
                raise FileNotFoundError("找不到模型权重文件")
        return model

    def _init_lightweight_model(self, num_classes, embedding_size):
        """初始化轻量级模型"""
        lightweight_model = LightweightCowModel(num_classes, embedding_size)
        self.backbone = lightweight_model.backbone
        self.embedding_layer = lightweight_model.embedding_layer
        self.actual_model_name = lightweight_model.actual_model_name
        self.num_classes = num_classes

    def _setup_embedding_layer(self):
        """设置嵌入层"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )

    def forward(self, x):
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.mean(dim=[2, 3])  # 全局平均池化
        embeddings = self.embedding_layer(features)
        # logits = self.classifier(embeddings) if self.classifier else None
        # return embeddings, logits
        return embeddings
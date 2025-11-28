import os
import json
import torch
import torch.nn as nn
import timm
from pathlib import Path
import logging
import torchvision.models as models
import numpy as np


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

    def __init__(self, model_name='MegaDescriptor-L-384', num_classes=None,
                 embedding_size=1024, use_lightweight=False, use_hf_snapshot=True):
        super().__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.use_lightweight = use_lightweight
        self.logger = logging.getLogger(__name__)
        self.img_size = 384

        if use_lightweight:
            self.logger.info("使用轻量级ResNet50模型")
            self._init_lightweight_model(num_classes, embedding_size)
            return

        try:
            self.logger.info(f"模型初始化")
            self.backbone, model_config = self.load_local_megadescriptor_model(use_hf_snapshot=use_hf_snapshot)
            self.actual_model_name = model_config['architecture']
            self.img_size = model_config.get('img_size', 384)
            self._setup_embedding_layer()
            self.num_classes = num_classes
            self.logger.info(f"成功初始化模型")

        except Exception as e:
            self.logger.error(f"MegaDescriptor加载失败: {e}")
            self.logger.info("切换到轻量级模型")
            self._init_lightweight_model(num_classes, embedding_size)

    def load_local_megadescriptor_model(self, use_hf_snapshot=True):
        """加载MegaDescriptor-L-384模型"""
        # 根据模型名称确定对应的 HuggingFace 仓库、默认backbone和输入尺寸
        if self.model_name in ['MegaDescriptor-B-224', 'MegaDescriptor-B', 'MegaDescriptor_B_224']:
            repo_id = "BVRA/MegaDescriptor-B-224"
            default_arch = "swin_base_patch4_window7_224"
            img_size = 224
            is_b_model = True
        else:
            # 默认保持向后兼容 MegaDescriptor-L-384
            repo_id = "BVRA/MegaDescriptor-L-384"
            default_arch = "swinv2_large_window12to24_192to384"
            img_size = 384
            is_b_model = False

        # 如果禁用 HuggingFace snapshot，则仅使用本地构建的 backbone 结构，不做任何远程访问
        if not use_hf_snapshot:
            config = {'architecture': default_arch, 'img_size': img_size}
            model = timm.create_model(
                default_arch,
                pretrained=False,
                num_classes=0,
                global_pool='avg',
                img_size=img_size
            )
            return model, config

        # 首选使用 snapshot_download 获取精确的快照目录
        preferred_cache_root = Path("./model_cache")
        model_dir = None
        try:
            from huggingface_hub import snapshot_download
            self.logger.info(f"通过 snapshot_download 获取 {repo_id} 的本地副本")
            snapshot_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(preferred_cache_root),
                local_dir_use_symlinks=False
            )
            model_dir = Path(snapshot_path)
        except ImportError:
            self.logger.warning("未安装 huggingface_hub，跳过显式 snapshot_download，将尝试使用默认缓存目录。")
        except Exception as e:
            self.logger.warning(f"snapshot_download 失败: {e}，将尝试使用默认缓存目录或其他回退方案。")

        # 如果 snapshot_download 未能提供有效目录，再尝试旧的 cache 结构作为补充
        if model_dir is None or not model_dir.exists():
            org, name = repo_id.split("/")
            cache_subdir = f"models--{org}--{name}"

            possible_cache_roots = [
                preferred_cache_root,
                Path("../model_cache"),
                Path.home() / ".cache" / "huggingface" / "hub",
            ]

            possible_cache_dirs = []
            for root in possible_cache_roots:
                possible_cache_dirs.append(root / cache_subdir / "snapshots" / "main")
                possible_cache_dirs.append(root / cache_subdir / "snapshots")

            for cache_dir in possible_cache_dirs:
                if cache_dir.exists():
                    # 如果是snapshots目录，找最新的snapshot
                    if cache_dir.name == "snapshots":
                        snapshots = [d for d in cache_dir.iterdir() if d.is_dir()]
                        if snapshots:
                            model_dir = max(snapshots, key=lambda x: x.stat().st_mtime)
                            break
                    else:
                        model_dir = cache_dir
                        break

        # 如果找到本地缓存，从本地加载
        if model_dir and model_dir.exists():
            self.logger.info(f"从本地加载 {repo_id}: {model_dir}")

            # 读取配置
            config_path = model_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                model_name = config.get('architecture', default_arch)
            else:
                config = {'architecture': default_arch}
                model_name = config['architecture']

            # 创建模型
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool='avg',
                img_size=img_size
            )

            self.logger.info(f"使用backbone架构: {model_name} (img_size={img_size})")

            # 尝试加载权重
            weights_path = model_dir / "pytorch_model.bin"
            if weights_path.exists():
                self.logger.info(f"加载权重: {weights_path}")
                state_dict = torch.load(weights_path, map_location='cpu')
                # 如果state_dict有'model'键，提取它
                if isinstance(state_dict, dict) and 'model' in state_dict:
                    state_dict = state_dict['model']

                model_state = model.state_dict()
                filtered_state = {}
                mismatched_keys = []

                for k, v in state_dict.items():
                    if k not in model_state:
                        continue
                    if model_state[k].shape != v.shape:
                        mismatched_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                        continue
                    filtered_state[k] = v

                if mismatched_keys:
                    self.logger.warning(f"发现 {len(mismatched_keys)} 个shape不匹配的权重，将跳过这些权重加载")

                model.load_state_dict(filtered_state, strict=False)
            else:
                safetensors_path = model_dir / "model.safetensors"
                if safetensors_path.exists():
                    self.logger.info(f"加载权重: {safetensors_path}")
                    from safetensors.torch import load_file
                    state_dict = load_file(str(safetensors_path))

                    model_state = model.state_dict()
                    filtered_state = {}
                    mismatched_keys = []

                    for k, v in state_dict.items():
                        if k not in model_state:
                            continue
                        if model_state[k].shape != v.shape:
                            mismatched_keys.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                            continue
                        filtered_state[k] = v

                    if mismatched_keys:
                        self.logger.warning(f"发现 {len(mismatched_keys)} 个shape不匹配的权重，将跳过这些权重加载")

                    model.load_state_dict(filtered_state, strict=False)
                else:
                    self.logger.warning("未找到预训练权重，使用随机初始化")
            config['img_size'] = img_size
        else:
            # 对于 B-224，我们只接受 BVRA 官方权重，不再回退到 timm 的预训练权重
            if is_b_model:
                raise RuntimeError(f"无法在本地找到 {repo_id} 的权重快照，且 snapshot_download 失败。")

            # 对于其他模型（如 L-384），保留之前的回退逻辑：
            # 从HuggingFace Hub加载（将自动缓存到本地），使用 timm 的预训练权重
            self.logger.info(f"从HuggingFace Hub加载 {repo_id}")
            try:
                from transformers import AutoConfig

                # 尝试使用transformers加载配置
                hf_config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
                model_name = getattr(hf_config, 'backbone', default_arch)

                # 使用timm创建backbone
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    global_pool='avg',
                    img_size=img_size
                )
                config = {'architecture': model_name, 'img_size': img_size}
            except Exception as e:
                self.logger.warning(f"无法从HuggingFace加载 {repo_id}: {e}")
                # 使用默认配置 + ImageNet 预训练权重 作为退路
                config = {'architecture': default_arch, 'img_size': img_size}
                model_name = config['architecture']
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    num_classes=0,
                    global_pool='avg',
                    img_size=img_size
                )

        return model, config

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
            # MegaDescriptor-L-384 使用 384x384 输入
            img_size = getattr(self, "img_size", 384)
            dummy_input = torch.randn(1, 3, img_size, img_size)
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
        return embeddings
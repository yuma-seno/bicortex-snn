import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

class VisualEncoder:
    def __init__(self, device='cpu'):
        """
        MobileNetV3-Small based Visual Encoder for SNN.
        Extracts 576-dimensional feature vector from images.
        """
        self.device = device
        
        # 1. Load Pre-trained Model
        # MobileNetV3 Small: 軽量で高速、エッジデバイス向き
        print("Loading MobileNetV3-Small...")
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        self.model = models.mobilenet_v3_small(weights=weights)
        
        # 2. Modifying Architecture
        # 分類ヘッド(Classifier)は不要なので無効化し、特徴抽出部(Features)のみ使う
        # MobileNetV3-Small の avgpool 直後の出力は [Batch, 576, 1, 1]
        self.model.classifier = nn.Identity() 
        
        self.model.to(self.device)
        self.model.eval() # 推論モード固定
        
        # 3. Preprocessing Pipeline
        # ImageNetの学習時に使われた正規化パラメータ
        self.preprocess = weights.transforms()
        
        print("Visual Encoder Ready. Output Dimension: 576")

    def encode(self, image: Image.Image) -> np.ndarray:
        """
        PIL Image -> 576-dim Feature Vector (Current for SNN)
        """
        # 前処理 (Resize, CenterCrop, Normalize)
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward pass
            # features(conv) -> avgpool -> flatten
            x = self.model.features(img_tensor)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            
            # 出力を取得 (Batchサイズ1なので [0] を返す)
            # 値の範囲は ReLU後なので 0.0 ~ ∞ だが、通常 0.0 ~ 6.0 程度
            feature_vector = x.cpu().numpy()[0]
            
        return feature_vector

    def get_output_dim(self):
        return 576
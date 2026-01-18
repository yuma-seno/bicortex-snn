import sys
import os
import numpy as np
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.visual import VisualEncoder

def test_visual_encoder():
    print("=== Testing Visual Encoder ===")
    
    # 1. Encoder 初期化
    encoder = VisualEncoder()
    dim = encoder.get_output_dim()
    print(f"Expected Dimensions: {dim}")
    
    # 2. ダミー画像の作成 (真っ赤な画像)
    print("\nCreating dummy image (Red)...")
    img_red = Image.new('RGB', (128, 128), color='red')
    
    # 3. エンコード実行
    features_red = encoder.encode(img_red)
    
    print(f"Output Shape: {features_red.shape}")
    print(f"Output Stats: Min={features_red.min():.4f}, Max={features_red.max():.4f}, Mean={features_red.mean():.4f}")
    
    # 4. ダミー画像の作成 (真っ青な画像)
    print("\nCreating dummy image (Blue)...")
    img_blue = Image.new('RGB', (128, 128), color='blue')
    features_blue = encoder.encode(img_blue)
    
    # 5. 比較 (赤と青で違う特徴が出ているか)
    diff = np.linalg.norm(features_red - features_blue)
    print(f"\nDistance between Red and Blue features: {diff:.4f}")
    
    if features_red.shape[0] == 576 and diff > 0.0:
        print("\n✅ Visual Encoder Test Passed!")
    else:
        print("\n❌ Test Failed.")

if __name__ == "__main__":
    test_visual_encoder()
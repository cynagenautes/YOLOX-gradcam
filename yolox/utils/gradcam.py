import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model: YOLOXモデル
            target_layer: 特徴マップを抽出する対象レイヤー
                         (通常はバックボーンの最終層またはFPNレイヤー)
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.features = None
        
        # フックを登録
        self._register_hooks()
        
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.features = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        # ターゲットレイヤーにフックを追加
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                target_found = True
                break
        
        if not target_found:
            raise ValueError(f"指定されたレイヤー '{self.target_layer}' が見つかりません")
    
    def remove_hooks(self):
        """登録されたフックを削除"""
        for hook in self.hooks:
            hook.remove()
    
    def __call__(self, inputs, class_idx=None, box_idx=0, img_info=None):
        """
        Grad-CAMを計算
        Args:
            inputs: モデルへの入力テンソル
            class_idx: クラスインデックス (Noneの場合は最大スコアのクラスを使用)
            box_idx: ボックスのインデックス (複数検出時)
            img_info: 画像変換情報 (リサイズ比率など)
        Returns:
            cam: Grad-CAMヒートマップ (元の画像サイズにリサイズ済み)
        """
        # モデルの勾配をクリア
        self.model.zero_grad()
        
        # 推論実行
        with torch.enable_grad():
            outputs = self.model(inputs)
        
        # 出力形式に応じて処理を変える必要がある場合がある
        # YOLOXの出力形式に合わせて適切に対応
        if class_idx is None:
            # スコア最大のクラスを選択
            scores = outputs[0][:, 4] * outputs[0][:, 5]
            if len(scores) == 0:
                return None
            idx = torch.argmax(scores)
            class_idx = int(outputs[0][idx, 6].item())
            box_idx = idx
            
        # 対象のボックスとクラススコアを取得
        objectness = outputs[0][box_idx, 4]  # objectness score
        class_scores = outputs[0][box_idx, class_idx+5]
        target = objectness * class_scores

        # 勾配を計算
        target.backward()
        
        # 勾配の重みを計算
        gradients = self.gradients.detach()
        features = self.features.detach()
        
        # グローバルプーリング
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
        
        # 重み付き和を計算
        cam = torch.sum(weights * features, dim=1, keepdim=True)
        
        # ReLUを適用
        cam = F.relu(cam)
        
        # CAM特徴マップを入力サイズに拡大
        cam = F.interpolate(cam, size=inputs.shape[2:], mode='bilinear', align_corners=False)
        
        # 正規化
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min).div(cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
        
        # numpy配列に変換
        cam = cam[0, 0].cpu().numpy()
        
        return cam

def apply_gradcam(img, cam, bbox=None, img_info=None):
    """
    Grad-CAMヒートマップを画像に適用
    Args:
        img: 元の画像 (OpenCV BGR形式)
        cam: Grad-CAMヒートマップ
        bbox: バウンディングボックス [x0, y0, x1, y1] (Noneの場合は画像全体)
        img_info: 画像変換情報 (リサイズ比率など)
    Returns:
        視覚化された画像
    """
    height, width = img.shape[:2]
    
    # パディング情報が提供されている場合
    if img_info is not None and "ratio" in img_info:
        ratio = img_info["ratio"]
        # モデル入力サイズ
        input_h, input_w = int(height * ratio), int(width * ratio)
        
        # パディング量を計算
        pad_h = img_info["test_size"][0] - input_h
        pad_w = img_info["test_size"][1] - input_w
        
        # パディングを除いたCAMを取得
        if pad_h > 0 or pad_w > 0:
            # CAMから有効部分のみ抽出（パディング部分を除去）
            cam = cam[:input_h, :input_w]
    
    # ヒートマップをリサイズして元の画像サイズに合わせる
    heatmap = cv2.resize(cam, (width, height))
    
    # ヒートマップをカラーマップに変換
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # 結果画像のコピーを作成
    result = img.copy()
    
    if bbox is not None:
        # バウンディングボックス内だけにヒートマップを適用
        x0, y0, x1, y1 = [int(coord) for coord in bbox]
        # 座標が画像範囲内にあることを確認
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        
        # バウンディングボックス内のマスクを作成
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 1
        
        # マスクを適用したヒートマップを元画像に重ね合わせる
        masked_heatmap = heatmap * mask[:, :, np.newaxis]
        result = cv2.addWeighted(result, 1.0, masked_heatmap, 0.4, 0)
    else:
        # 画像全体にヒートマップを重ね合わせる（従来の動作）
        result = cv2.addWeighted(result, 1.0, heatmap, 0.4, 0)
    
    return result
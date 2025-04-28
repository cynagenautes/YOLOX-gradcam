#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch
from torchinfo import summary

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from yolox.utils.gradcam import GradCAM, apply_gradcam

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # Grad-CAM関連の引数を追加
    parser.add_argument(
        "--gradcam",
        default=False,
        action="store_true",
        help="Enable Grad-CAM visualization",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default="backbone.backbone.dark5.2.conv1",
        help="Target layer for Grad-CAM visualization",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
        use_gradcam=False,
        target_layer=None,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        
        # Grad-CAM関連
        self.use_gradcam = use_gradcam
        self.target_layer = target_layer
        self.gradcam = None
        
        if use_gradcam and target_layer is not None:
            self.gradcam = GradCAM(model, target_layer)
        
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img_info["test_size"] = self.test_size

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        if self.use_gradcam:
            # Grad-CAM用に勾配を有効化
            with torch.set_grad_enabled(True):
                outputs = self.model(img)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
                # Grad-CAMを計算するために入力と出力を保存
                self.last_input = img
                self.last_outputs = outputs
        else:
            with torch.no_grad():
                t0 = time.time()
                outputs = self.model(img)
                if self.decoder is not None:
                    outputs = self.decoder(outputs, dtype=outputs.type())
                outputs = postprocess(
                    outputs, self.num_classes, self.confthre,
                    self.nmsthre, class_agnostic=True
                )
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        
        return outputs, img_info

    # 複数オブジェクトへのGrad-CAM適用（バウンディングボックス毎に個別画像を生成）
    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        
        if output is None:
            return img
            
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        
        # 通常の可視化（全検出結果を含む）
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        
        # Grad-CAM可視化用の結果リスト
        gradcam_images = []
        
        # Grad-CAM可視化（各オブジェクトに個別に適用）
        if self.use_gradcam and len(scores) > 0 and self.gradcam is not None:
            # スコア順にソート
            sorted_indices = torch.argsort(scores, descending=True)
            
            # スコアが閾値以上のオブジェクトを処理
            valid_indices = [idx for idx in sorted_indices if scores[idx] >= cls_conf]
            
            for i, idx in enumerate(valid_indices):
                class_idx = int(cls[idx].item())
                bbox = bboxes[idx].tolist()
                score = scores[idx].item()

                # 元画像のコピーを作成（個別の結果表示用）
                individual_img = img.copy()
                
                # このオブジェクトだけの通常検出結果を描画
                bbox_tensor = bboxes[idx:idx+1]  # 1つのボックスだけを含むテンソル
                score_tensor = scores[idx:idx+1]  # 1つのスコアだけを含むテンソル
                cls_tensor = cls[idx:idx+1]  # 1つのクラスだけを含むテンソル
                
                # 個別のオブジェクト検出結果を描画
                individual_img = vis(individual_img, bbox_tensor, score_tensor, cls_tensor, 0, self.cls_names)
                
                # Grad-CAMの計算
                img_info["test_size"] = self.test_size
                cam = self.gradcam(self.last_input, class_idx=class_idx, box_idx=idx, img_info=img_info)
                
                if cam is not None:
                    # この特定のオブジェクトだけにGrad-CAMを適用
                    gradcam_img = apply_gradcam(individual_img, cam, bbox, img_info)
                    
                    # # 画像に追加情報を表示
                    class_name = self.cls_names[class_idx]
                    # label_text = f"Class: {class_name}, Score: {score:.2f}"
                    # cv2.putText(
                    #     gradcam_img,
                    #     label_text,
                    #     (10, 30),  # 左上に表示
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8,
                    #     (0, 255, 0),
                    #     2
                    # )
                    
                    # 結果を保存
                    gradcam_images.append({
                        "image": gradcam_img,
                        "class_id": class_idx,
                        "class_name": class_name,
                        "score": score,
                        "bbox": bbox
                    })
        
        # 全体の検出結果と各オブジェクト個別のGrad-CAM結果を返す
        return {"full_image": vis_res, "gradcam_images": gradcam_images}


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result = predictor.visual(outputs[0], img_info, predictor.confthre)
        
        # 全体の検出結果画像
        full_image = result["full_image"]
        
        # 各オブジェクト個別のGrad-CAM画像
        gradcam_images = result.get("gradcam_images", [])
        
        if save_result:
            # 保存用フォルダ作成
            timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            base_name = os.path.basename(image_name).split(".")[0]
            
            save_folder = os.path.join(vis_folder, f"{timestamp}_{base_name}")
            os.makedirs(save_folder, exist_ok=True)
            
            # 全体の検出結果画像を保存
            full_image_path = os.path.join(save_folder, f"{base_name}_full.jpg")
            cv2.imwrite(full_image_path, full_image)
            logger.info(f"保存: {full_image_path}")
            
            # 各オブジェクト個別のGrad-CAM画像を保存
            for i, item in enumerate(gradcam_images):
                obj_image = item["image"]
                class_name = item["class_name"]
                score = item["score"]
                
                # クラス名とスコアを含むファイル名で保存
                obj_image_path = os.path.join(
                    save_folder, 
                    f"{base_name}_{i}_{class_name}_{score:.2f}.jpg"
                )
                cv2.imwrite(obj_image_path, obj_image)
                logger.info(f"保存: {obj_image_path}")


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    summary(model, (1, 3, 416, 416), device=args.device)
    print([name for name, _ in model.named_modules()])
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
        use_gradcam=args.gradcam, target_layer=args.target_layer
    )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)

# Ultralytics YOLO 🚀, GPL-3.0 license

import torch
import serial.tools.list_ports
import argparse
import sys
import os
import json

# 添加搜索路径
yolov8_project_path = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(
                        os.path.abspath(__file__)))))))
sys.path.append(yolov8_project_path)

from yolov8.ultralytics.yolo.engine.predictor import BasePredictor
from yolov8.ultralytics.yolo.engine.results import Results
from yolov8.ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from yolov8.ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box


class Command_return_body:
    def __init__(self):
        self.source_name = None


# 信息传递对象
clt_body = Command_return_body()


class DetectionPredictor(BasePredictor):  # 继承BasePredictor类
    def __int__(self):
        self.ser = ser

    def process_store_path(self):
        source_file_name = clt_body.source_name
        handle_file_name = self.save_dir.parts[-1]
        try:
            with open(os.path.join(os.path.dirname(self.save_dir), 'data.json'), 'r') as file:
                json_data_from_file = file.read()
            # 反序列化 JSON 字符串回字典
            data_from_file = json.loads(json_data_from_file)
        except FileNotFoundError as e:
            data_from_file = {}

        data_from_file[source_file_name] = handle_file_name

        # 序列化字典为 JSON 字符串
        json_data = json.dumps(data_from_file, indent=4)

        with open(os.path.join(os.path.dirname(self.save_dir), 'data.json'), 'w') as file:
            file.write(json_data)


    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

    def serial_send(self):
        # Process predictions
        """"
        判据0：不可靠
        if len(pred[0]) == 0:
            data = '2'  # 绿灯
            ser.write(data.encode('utf-8'))
        else:
            data = '1'  # 红灯
            ser.write(data.encode('utf-8'))
        """

        all_coord = self.results[0]  # 父类属性
        all_coord = all_coord.boxes.boxes  # 提取结果中的boxes属性
        locs = []
        locs_right = []
        for coord in all_coord:  # 计算所有液滴的x轴坐标，并存放在locs中
            location_x = (coord[0] + coord[2]) / 2
            locs.append(location_x)
            locs_right.append(coord[2])

        # if len(pred[0]) != 0:
        # if any(_ < 300 for _ in locs):  # any() 函数用于判断给定的可迭代参数 iterable 是否全部为 False,则返回 False,如果有一个为 True,则返回 True
        if any(_ < 300 for _ in locs_right):
            data = '1'  # 红灯，检测到液滴
            self.ser.write(data.encode('utf-8'))
        else:
            data = '2'  # 绿灯
            # ser.write(data.encode('utf-8'))

        # End of my Procession


def predict(opts, cfg=DEFAULT_CFG, use_python=False):
    model = opts.yolo_weights
    source = opts.source

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, default="yolov8x.pt", help='model.pt path(s)')
    parser.add_argument('--source', type=str, default="/home/elvis/shareFolder/MOT17.mp4", help='mp4 source')
    parser.add_argument('--upload_dir', type=str,
                        default="/home/elvis/shareFolder/MOTTrack_Git/yolov8/flask_remote_control", help='upload dir')

    opt = parser.parse_args()
    clt_body.source_name = opt.source.split("/")[-1]

    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(sys.path)
    print(os.getcwd())

    # 更改当前工作目录为上传文件的目录
    os.chdir(opt.upload_dir)
    predict(opt)

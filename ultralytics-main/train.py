from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'yolov11.yaml')  # 此处以 m 为例，只需写yolov11m即可定位到m模型
    # model.load('yolov11m.pt') # 是否加载预训练权重
    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=100,
                single_cls=False,  # 多类别设置False
                batch=16,
                workers=1,
                device='0',
                amp=False
                )

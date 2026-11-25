from ultralytics import YOLO

if __name__ == "__main__":

    # 1. 사전 학습된 YOLO11n 모델 불러오기
    model = YOLO("yolo11n.pt")

    # 2. 은행나무 객체 탐지 모델 학습
    model.train(
        data=r"C:\Users\user\Documents\TEAM-09-Final-Project\Ginkgo Tree Detection.v1-ginkgo_yolo_v1.yolov11 2\Ginkgo Tree Detection.v1-ginkgo_yolo_v1.yolov11\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        project="runs",
        name="tree_detector_n100"
    )

    print("학습 완료!")
from ultralytics import YOLO

if __name__ == "__main__":

    # 최종 선정 모델 불러오기
    model = YOLO("runs/detect/runs/tree_detector_n100/weights/best.pt")

    # Test 데이터셋 성능 평가
    metrics = model.val(
        data=r"C:\Users\user\Documents\TEAM-09-Final-Project\Ginkgo Tree Detection.v1-ginkgo_yolo_v1.yolov11 2\Ginkgo Tree Detection.v1-ginkgo_yolo_v1.yolov11\data.yaml",
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        plots=True,
        name="tree_test_final"
    )

    print("\n===== 최종 Test 성능 평가 결과 =====")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
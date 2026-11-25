from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("runs/detect/runs/tree_detector/weights/best.pt")

    model.predict(
        source="sample.jpg",
        conf=0.25,
        save=True,
        show=True
    )
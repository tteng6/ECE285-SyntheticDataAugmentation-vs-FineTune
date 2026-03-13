from ultralytics import YOLO
def run():
    model = YOLO("runs/detect/baseline3/weights/best.pt")

    model.train(
        data=r"D:\UCSD_courses\ECE285\project\yolo\config\bdd_finetune.yaml",
        epochs=120,
        imgsz=640,
        batch=16,
        name="bdd_finetune"
    )

if __name__ == "__main__":
    run()
from ultralytics import YOLO
def run():
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"D:\UCSD_courses\ECE285\project\yolo\config\fg_preserved.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="fg_preserved"
    )

if __name__ == "__main__":
    run()
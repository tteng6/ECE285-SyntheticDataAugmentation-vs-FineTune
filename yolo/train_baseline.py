from ultralytics import YOLO

def run():
    # Load a COCO-pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 example dataset for 100 epochs
    model.train(
        data=r"D:\UCSD_courses\ECE285\project\yolo\config\baseline.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name='baseline')

if __name__ == "__main__":
    run()
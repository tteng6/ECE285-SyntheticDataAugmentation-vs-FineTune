from pathlib import Path
import random
import shutil

# reproducibility
SEED = 285
random.seed(SEED)

IMG_DIR = Path(r"D:\UCSD_courses\ECE285\project\data\bdd_sampled\images")
LBL_DIR = Path(r"D:\UCSD_courses\ECE285\project\data\bdd_sampled\labels")

OUT_IMG = Path(r"D:\UCSD_courses\ECE285\project\data\bdd_yolo\image")
OUT_LBL = Path(r"D:\UCSD_courses\ECE285\project\data\bdd_yolo\labels")

# create folders
for split in ["train", "val"]:
    (OUT_IMG / split).mkdir(parents=True, exist_ok=True)
    (OUT_LBL / split).mkdir(parents=True, exist_ok=True)

# collect images
images = list(IMG_DIR.glob("*.jpg"))

print("Total images:", len(images))

# shuffle reproducibly
random.shuffle(images)

# split
train_imgs = images[:400]
val_imgs = images[400:]

print("Train:", len(train_imgs))
print("Val:", len(val_imgs))

def copy_set(img_list, split):
    for img in img_list:
        label = LBL_DIR / (img.stem + ".txt")

        shutil.copy2(img, OUT_IMG / split / img.name)
        shutil.copy2(label, OUT_LBL / split / label.name)

copy_set(train_imgs, "train")
copy_set(val_imgs, "val")

print("Done.")
import os
import shutil
import cv2
from sklearn.model_selection import train_test_split

def preprocess_dataset(
    raw_root="data/raw",
    processed_root="data/processed",
    img_size=(224, 224),
    train_ratio=0.80,
    val_ratio=0.10
):
    classes = ["Cat", "Dog"]

    # Check folders exist
    for cls in classes:
        if not os.path.isdir(os.path.join(raw_root, cls)):
            raise FileNotFoundError(f"Missing folder: {raw_root}/{cls}")

    # Create target structure
    for split in ["train", "val", "test"]:
        for cls in ["cat", "dog"]:
            os.makedirs(os.path.join(processed_root, split, cls), exist_ok=True)

    for cls in classes:
        src_dir = os.path.join(raw_root, cls)
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]

        print(f"{cls}: {len(images)} images found")

        train_imgs, temp_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        val_imgs, test_imgs   = train_test_split(temp_imgs,   train_size=val_ratio/(1-train_ratio), random_state=42)

        print(f"  → train: {len(train_imgs)}, val: {len(val_imgs)}, test: {len(test_imgs)}")

        def copy_resize(src_names, dest_folder):
            for fname in src_names:
                src_path = os.path.join(src_dir, fname)
                dst_path = os.path.join(dest_folder, fname)

                img = cv2.imread(src_path)
                if img is None:
                    print(f"Cannot read: {src_path}")
                    continue

                img = cv2.resize(img, img_size)
                cv2.imwrite(dst_path, img)

        copy_resize(train_imgs, os.path.join(processed_root, "train", cls.lower()))
        copy_resize(val_imgs,   os.path.join(processed_root, "val",   cls.lower()))
        copy_resize(test_imgs,  os.path.join(processed_root, "test",  cls.lower()))

    print("\nPreprocessing finished.")

if __name__ == "__main__":
    preprocess_dataset()
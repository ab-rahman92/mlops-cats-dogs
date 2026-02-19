import os
import shutil
import pytest
from src.data_preprocess import preprocess_dataset
from PIL import Image

# Small temporary directory for testing
TEST_RAW_DIR = "data/test_raw"
TEST_PROCESSED_DIR = "data/test_processed"

@pytest.fixture(scope="module")
def setup_test_data():
    """Create valid small fake images so cv2 can read them"""
    os.makedirs(TEST_RAW_DIR, exist_ok=True)
    
    classes = ["Cat", "Dog"]
    for cls in classes:
        class_dir = os.path.join(TEST_RAW_DIR, cls)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(20):  # enough to avoid split empty-set error
            filename = f"img_{cls.lower()}_{i:03d}.jpg"
            path = os.path.join(class_dir, filename)
            
            # Create a tiny valid image (e.g. 10x10 red square)
            img = Image.new('RGB', (10, 10), color='red' if cls == "Cat" else 'blue')
            img.save(path, "JPEG")
    
    yield
    
    # Cleanup
    shutil.rmtree(TEST_RAW_DIR, ignore_errors=True)
    shutil.rmtree(TEST_PROCESSED_DIR, ignore_errors=True)

def test_preprocess_creates_correct_structure(setup_test_data):
    preprocess_dataset(
        raw_root=TEST_RAW_DIR,
        processed_root=TEST_PROCESSED_DIR,
        img_size=(32, 32),  # small for test speed
        train_ratio=0.6,          # ← smaller train
        val_ratio=0.2
    )
    
    expected_splits = ["train", "val", "test"]
    expected_classes = ["cat", "dog"]
    
    for split in expected_splits:
        for cls in expected_classes:
            path = os.path.join(TEST_PROCESSED_DIR, split, cls)
            assert os.path.isdir(path), f"Missing directory: {path}"
            # At least some files should be copied (approx split)
            files = os.listdir(path)
            assert len(files) > 0, f"No files in {path}"

def test_preprocess_skips_invalid_files(setup_test_data):
    # Add an invalid file
    invalid_file = os.path.join(TEST_RAW_DIR, "Cat", "not_an_image.txt")
    open(invalid_file, 'w').close()
    
    preprocess_dataset(
        raw_root=TEST_RAW_DIR,
        processed_root=TEST_PROCESSED_DIR,
        img_size=(32, 32)
    )
    
    # Check that .txt was NOT copied to processed folders
    for split in ["train", "val", "test"]:
        cat_dir = os.path.join(TEST_PROCESSED_DIR, split, "cat")
        files = os.listdir(cat_dir) if os.path.exists(cat_dir) else []
        assert "not_an_image.txt" not in files
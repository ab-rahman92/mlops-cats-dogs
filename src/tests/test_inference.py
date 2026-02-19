import os
import pytest
import numpy as np
from PIL import Image
import io
from src.inference import predict_image, load_model

# Path to a real small test image (you need at least one cat or dog image)
# For now, we'll create a dummy image in memory for pure unit test
# Later you can use a real one from data/processed/test/

@pytest.fixture
def dummy_cat_image_bytes():
    """Create a fake small red image (pretend it's a cat)"""
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

def test_load_model():
    """Test that model loads without crashing (assumes model exists)"""
    try:
        model = load_model()
        assert model is not None
        assert hasattr(model, 'predict')
    except FileNotFoundError:
        pytest.skip("Model file not found - run training first")

def test_predict_image_returns_valid_format(dummy_cat_image_bytes):
    result = predict_image(dummy_cat_image_bytes)
    
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert "confidence" in result
    assert "raw_probability_dog" in result
    
    assert result["predicted_class"] in ["cat", "dog"]
    assert "%" in result["confidence"]
    assert 0 <= result["raw_probability_dog"] <= 1

def test_predict_image_handles_invalid_input():
    with pytest.raises(Exception):  # or more specific like ValueError
        predict_image(b"not an image bytes")
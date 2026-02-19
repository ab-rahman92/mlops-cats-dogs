import os
import pytest
from unittest.mock import patch
from src.inference import predict_image, load_model

MODEL_PATH = "models/baseline_model.keras"  # or use os.getenv("MODEL_PATH", ...)

@pytest.fixture
def dummy_cat_image_bytes():
    from PIL import Image
    import io
    img = Image.new('RGB', (224, 224), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr.getvalue()

@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not present in CI")
def test_load_model():
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model file not found - skipping in CI or when model not trained")
    
    model = load_model()
    assert model is not None
    assert hasattr(model, 'predict')

@patch("src.inference.load_model")  # ← mock the load_model call
def test_predict_image_returns_valid_format(mock_load_model, dummy_cat_image_bytes):
    # Even without real model, we can test the shape/format of the output function
    # Option A: Let it run with dummy predict (mock returns fake array)
    mock_model = mock_load_model.return_value
    mock_model.predict.return_value = [[0.3]]  # fake dog prob < 0.5 → cat
    
    result = predict_image(dummy_cat_image_bytes)
    
    assert isinstance(result, dict)
    assert "predicted_class" in result
    assert result["predicted_class"] in ["cat", "dog"]
    assert "confidence" in result
    assert "%" in result["confidence"]
    assert "raw_probability_dog" in result
    assert 0 <= result["raw_probability_dog"] <= 1

def test_predict_image_handles_invalid_input():
    with pytest.raises(Exception):
        predict_image(b"garbage bytes not an image")
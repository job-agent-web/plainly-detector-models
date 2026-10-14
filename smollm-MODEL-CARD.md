---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM-135M
tags:
- text-classification
- ai-detection
- pytorch
- onnx
- transformers
language:
- en
metrics:
- accuracy
library_name: transformers
pipeline_tag: text-classification
---

# Joshfcooper/ai-text-detector-optimized

## Model Description

This is an ultra-optimized AI text detector based on SmolLM-135M, designed to distinguish between human-written and AI-generated text with high accuracy and blazing-fast inference speed.

## Key Features

- **High Accuracy**: 96.7% accuracy on test data
- **Ultra-Fast**: 103.1ms average inference time
- **Optimized Architecture**: Uses only 12 out of 30 transformer layers (60% compression)
- **Multiple Formats**: Available in both PyTorch (.pt) and ONNX (.onnx) formats
- **Production Ready**: Optimized for real-world deployment

## Model Architecture

- **Base Model**: HuggingFaceTB/SmolLM-135M
- **Compression**: 30 layers → 12 layers (selected layers: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
- **Feature Extraction**: 24 layer outputs → 13,824 features  
- **Classifier**: Linear probe with sigmoid activation
- **Parameters**: ~60% reduction from base model

## Usage

### ONNX Model (Recommended for Web/Production)

```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

# Load tokenizer and ONNX model
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
session = ort.InferenceSession("model.onnx")

def predict(text):
    # Tokenize
    tokens = tokenizer(text, truncation=True, padding='max_length', 
                      max_length=256, return_tensors="np")
    
    # Convert to int64 for ONNX
    feeds = {
        'input_ids': tokens['input_ids'].astype(np.int64),
        'attention_mask': tokens['attention_mask'].astype(np.int64)
    }
    
    # Run inference
    result = session.run(None, feeds)
    probability = result[0][0]
    
    # Interpret (model outputs inverted probabilities)
    human_prob = 1 - probability
    is_human = human_prob > 0.5
    
    return {
        'prediction': 'human' if is_human else 'ai',
        'human_probability': human_prob,
        'confidence': abs(human_prob - 0.5) * 2
    }

# Example usage
result = predict("Your text here...")
print(result)
```

### PyTorch Model

```python
import torch
from transformers import AutoTokenizer
import pickle

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
model = torch.load("pytorch_model.pt", map_location='cpu')
model.eval()

def predict_pytorch(text):
    tokens = tokenizer(text, truncation=True, padding='max_length', 
                      max_length=256, return_tensors="pt")
    
    with torch.no_grad():
        probability = model(tokens['input_ids'], tokens['attention_mask']).item()
    
    human_prob = 1 - probability  # Invert output
    is_human = human_prob > 0.5
    
    return {
        'prediction': 'human' if is_human else 'ai',
        'human_probability': human_prob,
        'confidence': abs(human_prob - 0.5) * 2
    }
```

## Performance Metrics

- **Accuracy**: 96.7%
- **Inference Time**: 103.1ms (average)
- **Model Size**: ~60% smaller than base model
- **Throughput**: ~10 predictions/second

## Training Details

The model was trained using a feature extraction approach:
1. Extract hidden states from 12 selected layers of SmolLM-135M
2. Mean pooling across sequence length with attention masking
3. Concatenate features from all layers (13,824 total features)
4. Train linear classifier with standardization
5. Export to ONNX for optimized inference

## Important Notes

⚠️ **Output Inversion**: This model outputs inverted probabilities. Use `1 - model_output` for human probability.

## Files Included

- `model.onnx`: ONNX model for web/production deployment
- `pytorch_model.pt`: PyTorch model for development
- `config.json`: Model configuration
- `deployment_config.json`: Deployment configuration with layer selection
- `scaler_params.json`: Feature standardization parameters

## License

Apache 2.0

## Citation

```bibtex
@misc{ai-text-detector-optimized,
  title={Ultra-Optimized AI Text Detector},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/Joshfcooper/ai-text-detector-optimized}
}
```

## Ethical Considerations

This model is designed to detect AI-generated text. Please use responsibly and be aware that:
- No detector is 100% accurate
- Results should be used as guidance, not definitive proof
- Consider privacy and consent when analyzing text
- Be aware of potential biases in training data

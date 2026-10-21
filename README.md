# 4B Chart Predictor (on-device)

Static web app that classifies a stock **chart image** (pixels only) as
`up`, `flat`, or `down` using a ResNet18 exported to ONNX. Inference runs
**entirely in the browser** with [ONNX Runtime Web](https://onnxruntime.ai/) —
no server, no upload, no API key.

Live: https://hrudu6212.github.io/4b-chart-predictor/

## Model

- Architecture: ResNet18 (ImageNet-style stem), dropout 0.25 + linear head (3 classes)
- Input: RGB chart image, stretched to 224×224, normalized with ImageNet mean/std
- Weights: `model.onnx` (44.7 MB, fp32, opset 17)
- Export verified against PyTorch: max abs logit difference 1.8e-7

Honest metrics on the held-out test split: accuracy 36.5% (majority-class
baseline 42.0%). This is a research artifact, **not** a trading signal.

## Why static

Free hosting with no cold starts and no VM to keep alive. The trade-off is that
the weights are public and there is no server-side API key — inference is
client-side.

## Run locally

```bash
python -m http.server 8080
# open http://localhost:8080
```

## Files

| file | purpose |
| --- | --- |
| `index.html` | UI |
| `app.js` | download, preprocess, ONNX inference, softmax |
| `model.onnx` | exported weights |
| `model_meta.json` | class names, normalization constants, input size |

## Disclaimer

Statistical chart-image estimate only; not financial advice or a guarantee.

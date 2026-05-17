## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

On first launch, the app downloads:

- **STL-10** (~2.6 GB) - natural photographs, similar to ImageNet
- **EuroSAT** (~89 MB) - satellite imagery, drastically different from ImageNet
- **MNIST** (~20 MB) - handwritten 0-9 digits, no overlap from ImageNet
- **ResNet18 ImageNet weights** (~45 MB)

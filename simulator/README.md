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

The Simulator has a total of 6 pages:
- an overview describing the simulator
- 4 pages (1 for each technique)
- 1 comparison page where the users can compare different techniques and under different conditions.

The idea of the simulator is to have the users become familiar with the idea each technique and see where one technique might perform better and under which conditions compared to another.

The most important page is the comparison page to the as described above.

The users can choose from 3 datasets to fine-tune/transfer learn on. Each dataset differs from ImageNet at different levels of intensity.

The simulator actually fine tunes the Resnet18 model, so the results seen are fairly valid and not just 'simulated'.


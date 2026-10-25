# MNIST digit recognition: MLP vs CNN vs Transformer

Three small neural networks trained on the MNIST **train** split and evaluated on the **test** split.

## Models
- **MLP**: one hidden layer (256 units). The simplest baseline; sees the image as a flat list of pixels.
- **CNN**: two convolution layers + a small classifier. Looks for local patterns (edges, curves).
- **Transformer (encoder)**: the image is cut into 4x4 patches, which are processed like words in a sentence.

## Augmentation
Small random rotations (±10°), shifts (10%) and zoom (90–110%) applied to **training images only**.
The test set is never augmented.

## Run
```bash
pip install -r requirements.txt
./run_all.sh            # all 6 experiments, then writes results/summary.md
# or a single one:
python train.py --model cnn --augment
pip install -r requirements.txt
python train_test.py
```
train_test.py is the final version of my code; train.py and run_all.sh were my first draft.

## Results
results/summary.md

## Notes
- Reported accuracy is from the **final epoch**, not the best epoch, so the test set is not used to pick a result.
- Per-epoch logs are in `results/*.json`.

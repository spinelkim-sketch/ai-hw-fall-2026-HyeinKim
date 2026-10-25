# MNIST digit recognition: MLP vs CNN vs Transformer
Referenced examples from Claude and Gemini. Ran the test in Google Colab.

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
| Model | Augmentation | Parameters | Test accuracy | Train time (s) |
|---|---|---|---|---|
| cnn | yes | 421,642 | 99.39% | 234.1 |
| cnn | no | 421,642 | 99.33% | 166.8 |
| mlp | yes | 203,530 | 98.26% | 216.1 |
| mlp | no | 203,530 | 98.23% | 142.5 |
| transformer | yes | 72,074 | 96.40% | 256.8 |
| transformer | no | 72,074 | 97.12% | 185.4 |

Test accuracy is the share of the 10,000 test images the model identified correctly. Higher is better, and 99% means about 100 mistakes.
Augmentation: yes/no shows whether the training images were randomly rotated and shifted. Compare the “yes” row with the “no” row for the same model. Did the number go up or down?
Parameters is the model’s size. Bigger models are not always more accurate.
Train time is how long training took.


## Notes
- Reported accuracy is from the **final epoch**, not the best epoch, so the test set is not used to pick a result.
- Per-epoch logs are in `results/*.json`.

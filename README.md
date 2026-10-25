# MNIST digit recognition: MLP vs CNN vs Transformer
Referenced examples from Claude and Gemini. Ran the test in Google Colab.

Three small neural networks trained on the MNIST (60,000 labeled digit images for studying) **train** split and evaluated on the 10,000 separate ones for the exam in **test** split.

## Models
- **MLP**: Looks at the picture as one long list of 784 pixel values (28*28), with no acknowledgement of which pixels are adjacent to each other. 
- **CNN**: two convolution layers + a small classifier. Looks for local patterns (edges, curves). This is like a magnifying glass over the image to spot edges and curves, and then combine them ("a loop on top+a straight line=9")
- **Transformer (encoder)**: the image is cut into 49 small tiles and let the tiles decide what the whole picture shows. It is like chatGPT in image tile context. 

## Augmentation
Small random rotations (±10°), shifts (10%) and zoom (90–110%) applied to **training images only**.
The test set is never augmented.

## Run
```bash
pip install -r requirements.txt
./run_all.sh           
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

Test accuracy: the share of the 10,000 test images the model identified correctly. Higher is better, and 99% means about 100 mistakes.
Augmentation: yes/no shows whether the training images were randomly rotated and shifted. 
Parameters: the model’s size. Bigger models are not always more accurate.
Train time: how long training took.

**Disclaimer

The Transformer explanation is a hypothesis. I didn’t test it, which is why the text says “a likely reason”. If you want to claim more, you could try training the Transformer for more epochs, for example python train.py --model transformer --epochs 30, and see whether the gap closes. That would also make a good extra slide.
Each result comes from a single training run. Differences of a few hundredths of a percent can change if you retrain with a different random seed, which is why the text calls them “too small to call a real improvement”.

**Key findings**

- **The CNN performed best** at 99.39% accuracy (about 61 mistakes out of 10,000 test images). It is designed to look for local shapes such as edges and curves, which suits handwritten digits.
- **Augmentation made almost no difference for the CNN and MLP.** Accuracy rose by only 0.06 and 0.03 percentage points, which is a handful of images and is too small to call a real improvement from a single run.
- **Augmentation hurt the Transformer**, lowering accuracy from 97.12% to 96.40%. A likely reason is that this small model, trained for only 10 epochs, had trouble learning from the more varied images.
- **The Transformer scored lowest overall** (96.40–97.12%). It is also the smallest model (72,074 parameters), and Transformers usually need more data or longer training to perform well on images.
- **Augmentation increased training time** for every model, by roughly 40 to 70 seconds, because each training image is randomly transformed on the fly.
- **Bigger was not better:** the MLP has about half the CNN's parameters, but its accuracy was about 1 percentage point lower.


## Notes
- Reported accuracy is from the **final epoch**, not the best epoch, so the test set is not used to pick a result.
- Per-epoch logs are in `results/*.json`.

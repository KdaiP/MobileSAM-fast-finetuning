<div align="center">

# MobileSAM-fast-finetuning

_✨ Finetune MobileSAM with Less Than 4GB RAM!  ✨_

</div>

MobileSAM-fast-finetuning is a training script designed for [MobileSAM](https://github.com/ChaoningZhang/MobileSAM), enabling efficient model finetuning on hardware with limited memory without using adapter.

The script has been tested on both Windows and Linux operating systems:

- Python version: 3.10

- PyTorch version: 2.1

## Installation

1. **PyTorch Installation**: Visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) to set up PyTorch on your system.

2. **Dependencies**: Once PyTorch is installed, install the required packages using the command:

```python
pip install -r requirements.txt
```

## Usage

### Preparing the Data

- **Training Data**: Place your training images (JPEG format) and corresponding masks (PNG format, same name as the images) in the `./datasets/train` directory.
- **Validation Data**: Place your validation images (JPEG format) and masks (PNG format, same name as the images) in the `./datasets/val` directory.

### Running the Training Script

Run `train.py`

By default, the checkpoint will be saved at  `./logs/`

Training setting (such as batch_size) could be modified in `./configs/mobileSAM.json`

## Inference

To use the finetuned MobileSAM model, simply replace the original MobileSAM checkpoint with the newly finetuned one. No additional configuration needed for a seamless transition!

## References

[MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
[Medical-SAM-Adapter](https://github.com/WuJunde/Medical-SAM-Adapter)
[SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
[MedSAM](https://github.com/bowang-lab/MedSAM)
[lightning-sam](https://github.com/luca-medeiros/lightning-sam)
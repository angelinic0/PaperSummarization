# PaperSummarization
Summarizing text from any picture/PDF using Tessaract OCR and Google Pegasus Summarization .

## Table of Contents
- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Maintainers](#maintainers)
- [Acknowledgements](#Acknowledgements)
- [License](#license)

## Background



## Installation for Ubuntu 18.04.4+

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements for
the paper summarization algorithm. 

```bash
sudo apt install poppler-utils
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev

pip install -r requirements.txt
```
## Usage

```python
python Lightning_main.py
```
or
```python
python main.py
```
Running will train a Variational Density Propagation model on the CIFAR10 benchmark dataset. Running the `Lighting_main.py` will utilize pytorch lighting and take advantage of Distributed Data Parallel for computational and time efficiency.  Running the `main.py` will run the original PyTorch code. `VDP_Layers.py` contains the custom density progagtion layers.

If your desire is to create your own custom network with addotional layers, please reference the `VDP_Layers.py` file where all the custom layers are implemented.

## Results

**Coming soon**


## Related Efforts

- [VDP for MNIST](https://github.com/angelinic0/VDP_MNIST/) - Variational Density Propagation for the MNIST Toy Dataset
- [VDP for CIFAR100](https://github.com/angelinic0/VDP_CIFAR100/) -  Variational Density Propagation for the CIFAR100 Toy Dataset
- ~~[VDP for IMAGENET](https://github.com/angelinic0/VDP_IMAGENET/) -  Variational Density Propagation with IMAGENET~~**Coming Soon**

## Maintainers

[@angelinic0](https://github.com/angelinic0).

## Acknowledgements

Thank you to [Rowan University](https://www.rowan.edu/) and the [United States Department of Education](https://www.ed.gov/) for hosting me for my PhD Research and for funding my education through the [GAANN Fellowship](https://www2.ed.gov/programs/gaann/index.html), respectively. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

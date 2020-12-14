# PaperSummarization
Summarizing text from any picture/PDF using Tessaract OCR and Google Pegasus Summarization .

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

Note that this is only available for Ubuntu

Before running the python script ensure that your path to Poppler and
Tesseract are the same. 

```python
python main.py --pdf_path ./compression.pdf
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

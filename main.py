# -*- coding: utf-8 -*-
"""Main file for running Tesseract OCR and Google PEGASUS summarization for a document

"""
import os
import re
import argparse
import pytesseract
from pdf2image import convert_from_path
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

parser = argparse.ArgumentParser(prog="Paper Summarizer")
# Utils
parser.add_argument('--model', type=str, default='google/pegasus-large', help='Path to already cleaned text')
parser.add_argument('--poppler_path', type=str, default='/bin/', help='Path to PDF')
parser.add_argument('--tesseract_path', type=str, default='/bin/tesseract', help='Path to Tesseract')
# Inputs
parser.add_argument('--pdf_path', type=str, default='./compression.pdf', help='Path to PDF')
parser.add_argument('--image_proc_list', type=str, default=None, help='Path to a list of images to process')
parser.add_argument('--text_from_image', type=str, default=None, help='Path to already translated text')
parser.add_argument('--cleaned_text', type=str, default='', help='Path to already cleaned text')
# Hypers for Pegasus
parser.add_argument('--max_length', type=int, default=400, help='Max Summary Length')
parser.add_argument('--min_length', type=int, default=100, help='Min Summary Length')
parser.add_argument('--do_sample', type=bool, default=True, help='Do sampling')
parser.add_argument('--temperature', type=float, default=3.0, help='Temp value')
parser.add_argument('--top_k', type=int, default=30, help='Top K')
parser.add_argument('--top_p', type=float, default=0.70, help='Top p')
parser.add_argument('--repetition_penalty', type=float, default=1.2, help='Repetition Penalty')
parser.add_argument('--length_penalty', type=int, default=5, help='Length Penalty')
parser.add_argument('--num_return_sequences', type=int, default=1, help='Number of return sequences')



def makeandmove(work, input):
    """

    :param work:
    :param input:
    :return:
    """
    paper_name = os.path.split(input)[-1].split('.')[0]
    dir = work + paper_name + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.system('cp ' + input + ' ' + dir)
    return dir


def pdf_to_image(pdf_path, poppler_path, work_dir):
    """

    :param pdf_path:
    :param poppler_path:
    :param proc_list:
    :return:
    """
    print('--- Converting PDF to Image ---')
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    image_proc_file = work_dir + 'image_proc_list.txt'
    if os.path.exists(image_proc_file):
        f1 = open(image_proc_file, "w")
    else:
        f1 = open(image_proc_file, "x")

    image_dir = work_dir + 'images/'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for i, img in enumerate(images):
        print('Saving Page ' + str(i) +
              ' as output' + str(i) + '.jpg to Working Directory')
        image_path = image_dir + 'output' + str(i) + '.jpg'
        img.save(image_path, 'JPEG')
        print(image_path, file=f1)
    f1.close()
    return image_proc_file


def perform_ocr(image_proc_file, work_dir):
    """

    :return:
    """

    print('--- Performing OCR ---')
    ocr_file = work_dir + 'ocr_text.txt'
    if os.path.exists(ocr_file):
        f1 = open(ocr_file, "w")
    else:
        f1 = open(ocr_file, "x")
    ocr_text = pytesseract.image_to_string(image_proc_file)
    print(ocr_text, file=f1)
    print('OCR Output Saved')
    f1.close()
    return ocr_text

def clean_text(ocr_text, work_dir):
    """

    :return:
    """
    print('--- Cleaning Text ---')
    clean_text = re.split('Introduction|INTRODUCTION', ocr_text)[1]
    clean_text = re.split('References|REFERENCES', clean_text)[0]
    clean_text = re.sub('[^\x00-\x7f]', '', clean_text)
    clean_text = re.sub('[\n\s?]{2,}', '\n', clean_text)
    clean_text = re.sub('\[.+\]', '', clean_text)
    clean_text = re.sub('\(.+\)', '', clean_text)
    clean_text_file = work_dir + 'clean_text.txt'
    if os.path.exists(clean_text_file):
        f1 = open(clean_text_file, "w")
    else:
        f1 = open(clean_text_file, "x")
    print(clean_text, file=f1)
    f1.close()
    return clean_text


def main():

    # parse Arguements
    params = parser.parse_args()
    # Define Tesseract bin
    pytesseract.pytesseract.tesseract_cmd = params.tesseract_path

    # Create working directory if doesn't exist
    work_dir = './workingDir/'
    if not os.path.exists(work_dir):
        print('Creating Working Directory')
        os.makedirs(work_dir)

    # For previously cleaned text
    if params.cleaned_text is not None:
        working_dir = makeandmove(work_dir, params.cleaned_text)
        if os.path.exists(params.cleaned_text):
            f1 = open(params.cleaned_text, 'r')
            cleaned_text = f1.read()
            f1.close()
        else:
            print('Something went wrong.')

    # For previously recognized text
    elif params.text_from_image is not None:
        working_dir = makeandmove(work_dir, params.text_from_image)
        if os.path.exists(params.text_from_image):
            f1 = open(params.text_from_image, 'r')
            ocr_text = f1.read()
            f1.close()
        else:
            print('Something went wrong.')
        cleaned_text = clean_text(ocr_text, working_dir)

    # For previously generated images
    elif params.image_proc_list is not None:
        working_dir = makeandmove(work_dir, params.image_proc_list)
        ocr_text = perform_ocr(params.image_proc_list, working_dir)
        cleaned_text = clean_text(ocr_text, work_dir)

    # For PDFs
    else:
        working_dir = makeandmove(work_dir, params.pdf_path)
        image_proc_file = pdf_to_image(params.pdf_path, params.poppler_path, working_dir)
        ocr_text = perform_ocr(image_proc_file, working_dir)
        cleaned_text = clean_text(ocr_text, working_dir)

    print('--- Summarizing Text ---')
    # download model
    model = PegasusForConditionalGeneration.from_pretrained(params.model)
    # download tokenizer
    tok = PegasusTokenizer.from_pretrained(params.model)
    batch = tok.prepare_seq2seq_batch(src_texts=[cleaned_text])
    # Hyperparameter Tuning

    gen = model.generate(
        **batch, max_length=params.max_length,
        min_length=params.min_length,
        do_sample=params.do_sample,
        temperature=params.temperature,
        top_k=params.top_k,
        top_p=params.top_p,
        repetition_penalty=params.repetition_penalty,
        length_penalty=params.length_penalty,
        num_return_sequences=params.num_return_sequences)

    summary = tok.batch_decode(gen, skip_special_tokens=True)

    summary_file = working_dir + 'summary.txt'
    if os.path.exists(summary_file):
        f1 = open(summary_file, "w")
    else:
        f1 = open(summary_file, "x")
    print(summary, file=f1)
    f1.close()
    print(summary)
    return None


if __name__ == '__main__':
    main()
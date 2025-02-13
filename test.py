import pytest
import pathlib
import platform
import nbimporter
import pickle as pkl
import pandas as pd
from pandas.testing import assert_frame_equal
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from pandas import DataFrame
from dataclasses import dataclass

from preprocessing import (extract_text, clean_text,
                           process_text, to_dataframe,
                           customize_tokenizer)


@dataclass
class Shared:
    nlp: English
    html_content: str
    noisy_target_text: str
    cleaned_target_text: str
    target_doc: Doc
    target_customized_doc: Doc
    target_df: DataFrame

   
@pytest.fixture(scope="session")
def shared():
    ext = ""
    if platform.system() == "Windows":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        ext = "_win"
    with open(f"test_utils/nlp{ext}.pkl", "rb") as pklfile:
        nlp = pkl.load(pklfile)
    with open("test_utils/test.html", encoding="utf8") as html_file:
        html_content = html_file.read()
    with open("test_utils/text.txt", encoding="utf8") as tfile:
        noisy_target_text = tfile.read()
    with open("test_utils/cleaned_text.txt", encoding="utf8") as tfile:
        cleaned_target_text = tfile.read()
    with open(f"test_utils/doc{ext}.pkl", "rb") as pklfile:
       target_doc = pkl.load(pklfile)
    with open(f"test_utils/customized_doc{ext}.pkl", "rb") as pklfile:
       target_customized_doc = pkl.load(pklfile)
    target_df = pd.read_csv("test_utils/df.csv", encoding="utf8").fillna('')
    return Shared(nlp, html_content, noisy_target_text, cleaned_target_text,
                  target_doc, target_customized_doc, target_df)


def assert_equal_docs(test_doc, target_doc):
    assert type(test_doc) == type(target_doc)
    assert len(test_doc) == len(target_doc)
    target_tokens = [token.text for token in target_doc]
    test_tokens = [token.text for token in test_doc]
    assert test_tokens == target_tokens
    target_ents = [ent.text for ent in target_doc.ents]
    test_ents = [ent.text for ent in test_doc.ents]
    assert test_ents == target_ents


def test_extract_text(shared):
    test_text = extract_text(shared.html_content)
    # assert test_text[:476] + test_text[-576:] == shared.noisy_target_text
    assert test_text == shared.noisy_target_text
   
   
def test_clean_text(shared):
    cleaned_test_text = clean_text(shared.noisy_target_text)
    assert cleaned_test_text == shared.cleaned_target_text

    
def test_process_text(shared):
    test_doc = process_text(shared.cleaned_target_text, shared.nlp)
    assert_equal_docs(test_doc, shared.target_doc)
    
   
def test_to_dataframe(shared):
    test_df = to_dataframe(shared.target_doc)
    assert_frame_equal(shared.target_df, test_df, check_dtype=False)


def test_customize_tokenizer(shared):
    test_customized_nlp = customize_tokenizer(shared.nlp)
    test_customized_doc = process_text(shared.cleaned_target_text, test_customized_nlp)
    assert_equal_docs(test_customized_doc, shared.target_customized_doc)

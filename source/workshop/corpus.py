"""Helper functions for corpus analysis."""
import os
import re

import pandas as pd
import nltk
from tqdm import tqdm
tqdm.pandas()

# Constants
GUTENBERG_LICENCE_REGEX = re.compile("\*{3}\s{0,2}END OF.+", flags=re.DOTALL)
GUTENBERG_BOILERPLATE_REGEX = re.compile("\A.+\*{3}\s{0,2}START OF[^*]+\*{3}", flags=re.DOTALL)
GUTENBERG_EMPHASIS_REGEX = re.compile("(?<=\W)[_=]{1,2}|[_=]{1,2}(?=\W)")

def import_corpus(
    corpus_dir: str,
    manifest_file: str="manifest.csv",
    to_lower: bool=True,
    clean_gutenberg: bool=True,
    tokenise_by: str="word",
    remove_punctuation: bool=False,
    to_stems: bool=False,
    stopwords: str|None=None,
    verbose: int=2
) -> pd.DataFrame:
    """
    Imports a corpus of texts and prepares them for text analysis. The function
    presumes you have all your texts together in a single folder, `corpus_dir`.
    In the corpus directory should also be a spreadsheet, in csv format, describing
    the texts. There must be a column called "file_name" that contains the file name
    for each text in the corpus.

    params:
        corpus_dir: a file path to a folder of .txt files with a .csv manifest
        manifest_file: the file name of the manifest
        to_lower: put texts in lowercase?
        clean_gutenberg: remove Gutenberg licence, boilerplate and emphasis?
        tokenise_by: either "word", "sent", or None (if you just want the raw text)
        remove_punctuation: remove punctuation from the tokens? 
        to_stems: apply stemmer to the tokens? (English only)
        stopwords: either `None`, in which case stopwords are ignored, or a str,
            giving the name of the NLTK stopword list to use (e.g. "english")
        verbose: 2=report import of each text, 1=report completion, 0=no output

    returns:
        corpus: a pd.DataFrame with metadata about the texts, and a column of tokens
    """

    # Create corpus data from from manifest file
    manifest_path = os.path.join(corpus_dir, manifest_file)
    if verbose > 0:
        print(f"Loading {manifest_path} ...")
    corpus = pd.read_csv(manifest_path)

    if "file_name" not in corpus.columns:
        raise ValueError(f"No 'file_name' column found in {manifest_path}.")
    
    # Configure importer
    importer = TextImporter(
        corpus_dir=corpus_dir,
        to_lower=to_lower,
        tokenise_by=tokenise_by,
        clean_gutenberg=clean_gutenberg,
        remove_punctuation=remove_punctuation,
        to_stems=to_stems,
        stopwords=stopwords,
        verbose=verbose
    )

    if verbose > 0:
        print(f"{corpus.shape[0]} texts found in {manifest_path}. Importing with the following settings:")
        print(importer)

    # Map onto file names
    if tokenise_by is None:
        corpus = corpus.assign(text=lambda df: df["file_name"].map(importer.import_text))
    else:
        corpus = corpus.assign(tokens=lambda df: df["file_name"].map(importer.import_text))
    if verbose > 0:
        print("Done!")

    return corpus

def _chunk_list(lst, n):
    """Split `lst` into chunks of size `n`. Return a list containing all
    the new, smaller lists. Drop the final chunk if it is less than
    n // 2 items long"""
    chunks = [lst[i:i+n] for i in range(0, len(lst), n)]
    
    if len(chunks[-1]) >= n // 2:  # If final chunk is long enough ...
        return chunks              # ... return the whole list of chunks
    else:                          # otherwise ...
        return chunks[:-1]         # ... return the list, except for the last chunk

def chunk_corpus(corpus: pd.DataFrame, chunk_size: int=2500, token_col: str="tokens", text_col: str="text"):
    """Given a data frame, `corpus`, that has a 'tokens' column, split each text
    into `chunk_size`-word chunks using 'tokens'. Discard the final chunk of each text if it
    is less than half of `chunk_size` (e.g. if the `chunk_size` is 5000,
    then drop the last chunk of each text if it is less than 2500 tokens long).

    params:
        corpus: A dataframe of texts; must have either a column of tokens or a
                column of full-text.
        chunk_size: The desired chunk size
        token_col: The name of the column containing the tokens, or None if the
                corpus has not been tokenised.
        text_col: The name of the column containing the full text; this is only required
                if the corpus is not already tokenised.
    """

    # If there is no tokens column, create it:
    if token_col not in corpus.columns:
        if text_col not in corpus.columns:
            raise ValueError(f"Neither '{token_col}' nor '{text_col}' exist. Is there a typo somewhere?")
        print(f"Creating '{token_col}' from '{text_col}' using `nltk.word_tokenize`...")
        corpus = corpus.assign(tokens = lambda df: df[text_col].progress_map(nltk.word_tokenize))

    # Temporarily rename text and tokens columns:
    corpus = corpus.rename({token_col: "tokens", text_col: "text"}, axis="columns")

    # Chunk
    corpus = (
        corpus
        .assign(
            # Use _chunk_list on the tokens column
            tokens = lambda df: df.tokens.map(lambda tokens: _chunk_list(tokens, chunk_size))
        )
        .explode("tokens") # Explode the tokens column, so that now we have one row per chunk
        .assign(
            # Recreate the 'text' column by joining together
            # all the tokens in each chunk
            text = lambda df: df.tokens.str.join(sep=" ")
        )
    )

    # Give the token column its original name back
    corpus = corpus.rename({"tokens": token_col, "text": text_col}, axis="columns")

    return corpus
    
class TextImporter():
    """Imports a text according to the desired settings."""

    def __init__(self, corpus_dir, to_lower, tokenise_by, clean_gutenberg, remove_punctuation, to_stems, stopwords, verbose):
        self.corpus_dir = corpus_dir
        self.to_lower = to_lower
        self.tokenise_by = tokenise_by
        self.clean_gutenberg = clean_gutenberg
        self.remove_punctuation = remove_punctuation
        self.to_stems = to_stems
        self.stopwords = stopwords
        self.verbose = verbose

    def __repr__(self):
        return f"TextImporter(corpus_dir='{self.corpus_dir}', to_lower={self.to_lower}, tokenise_by='{self.tokenise_by}', clean_gutenberg={self.clean_gutenberg} remove_punctuation={self.remove_punctuation}, to_stems={self.to_stems}, stopwords={self.stopwords})"

    def _read_text(self, file_path):
        full_path = os.path.join(self.corpus_dir, file_path)
        if self.verbose > 1:
            print(f"Reading {full_path} ...")
        with open(full_path, "rt") as text_file:
            return text_file.read()

    def _remove_gutenberg_text(self, text):
        if self.clean_gutenberg is True:
            text = GUTENBERG_BOILERPLATE_REGEX.sub("", text)
            text = GUTENBERG_LICENCE_REGEX.sub("", text)
            text = GUTENBERG_EMPHASIS_REGEX.sub("", text)
        return text

    def _tokenise(self, text):
        if self.tokenise_by == "word":
            return nltk.word_tokenize(text)
        elif self.tokenise_by == "sent":
            return nltk.sent_tokenize(text)
        else:
            raise ValueError(f"`tokenise_by` must be 'word' or 'sent', not {self.tokenise_by}")

    def _lower(self, text):
        if self.to_lower is True:
            return text.lower()
        elif self.to_lower is False:
            return text
        else:
            raise ValueError(f"`to_lower` must be `True` or `False`, not {self.to_lower}")

    def _remove_punctuation(self, tokens):
        word_rgx = re.compile("\w+")
        if self.remove_punctuation is True:
            return [t for t in tokens if word_rgx.match(t)]
        elif self.remove_punctuation is False:
            return tokens
        else:
            raise ValueError(f"`remove_punctuation` must be `True` or `False`, not {self.remove_punctuation}")

    def _stem(self, tokens):
        stemmer = nltk.LancasterStemmer()
        if self.to_stems is True and self.tokenise_by == "word":
            return [stemmer.stem(t) for t in tokens]
        elif self.to_stems is False:
            return tokens
        elif self.to_stems is True and self.tokenise_by != "word":
            raise ValueError(f"`to_stems` can only be `True` if `tokenise_by` is 'word': tokenise_by == {self.tokenise_by}")
        else:
            raise ValueError(f"`to_stems` must be `True` or `False`, not {self.to_stems}")

    def _remove_stopwords(self, tokens):
        if self.stopwords is None:
            return tokens
        else:
            return [t for t in tokens if token not in nltk.stopwords(self.stopwords)]

    def import_text(self, file_path):
        """Import the text at `path`, and apply desired preprocessing steps.
        Project Gutenberg boilerplate and licence removed automatically."""

        text = self._read_text(file_path)
        text = self._remove_gutenberg_text(text)
        text = self._lower(text)

        if self.tokenise_by is None:
            return text
        else:
            tokens = self._tokenise(text)
            tokens = self._remove_punctuation(tokens)
            tokens = self._remove_stopwords(tokens)
            tokens = self._stem(tokens)
            return tokens

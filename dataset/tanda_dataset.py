from __future__ import annotations

from typing import List, Any, Tuple, Dict, Union, Optional
from torch import Tensor

import json
import torch
import random as rd
import numpy as np

from dataset.base import Dataset
from dataset.utils import pad_and_tensorize, extract_sentences, extract_qas


Path = str
Tokenizer = Any
Paragraph = Tuple[List[str], List[str], List[int]]
Number = Union[int, float]


class TandaBatch:

    """
    This class is here only to be matched.
    """

    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
    ) -> None:
        self.X = X
        self.Y = Y

    def to(
        self,
        device : str,
    ) -> TandaBatch:
        self.X = (self.X[0].to(device), self.X[1].to(device))
        self.Y = self.Y.to(device)
        return self


class TandaDataset(Dataset):

    """
    A class to store questions and passages and deliver as is done in Tanda.
    Attributes:
        - paragraphs, a dictionnary where keys are paragraph titles and items are dictionnaries with questions, passages, answer_ids, tokenized, and eventually tokenized_questions and tokenized_passages
        - length, the number of questions in total
        - yield_order, a list of (paragraph_title, question_number) to build the batches
        - sep_token_id and pad_token_id, added during the tokenization
    """

    def __init__(
        self,
        paragraphs: Dict[str, Paragraph]
    ) -> None:
        """
        Initializes the TandaDataset with a list of (paragraphs)
        """
        self.length = 0
        self.paragraphs = {}
        self.pad_token_id = None
        self.sep_token_id = None
        for title, paragraph in paragraphs.items():
            self.paragraphs[title] = {}
            self.paragraphs[title]['questions'], self.paragraphs[title]['passages'], self.paragraphs[title]['answer_ids'] = paragraph
            self.paragraphs[title]['tokenized'] = False
            self.length += len(self.paragraphs[title]['questions'])
        self.build_yield_order()

    def __len__(self,) -> int:
        return self.length

    def split_paragraph_wise(
        self,
        p : float,
        seed : Optional[int] = None,
    ) -> Tuple[TandaDataset, TandaDataset]:
        """
        Splits the TandaDataset in two according to the proportion (p) passed.
        For reproducibility's sake, a (seed) can be passed too.
        """
        # For reproducibility
        rng = np.random.default_rng(seed)
        # Accumulators
        paragraphs_a, paragraphs_b = {}, {}
        # Paragraph-level loop
        for title, paragraph in self.paragraphs.items():
            # Copy passages and tokenized for both
            paragraphs_a[title] = {'passages' : paragraph['passages'] + [], 'tokenized' : paragraph['tokenized']}
            paragraphs_b[title] = {'passages' : paragraph['passages'] + [], 'tokenized' : paragraph['tokenized']}
            # Eventually copy tokenized passages
            if paragraph['tokenized']:
                paragraphs_a[title]['tokenized_passages'] = paragraph['tokenized_passages'] + []
                paragraphs_b[title]['tokenized_passages'] = paragraph['tokenized_passages'] + []
            # Prepare permutation for questions and answer_ids split
            permutation = [i for i in range(len(paragraph['questions']))]
            rng.shuffle(permutation)
            cutout = int(len(paragraph['questions']) * p)
            # Questions split
            questions = [paragraph['questions'][i] for i in permutation]
            paragraphs_a[title]['questions'], paragraphs_b[title]['questions'] = questions[:cutout], questions[cutout:]
            # Answer_ids split
            answer_ids = [paragraph['answer_ids'][i] for i in permutation]
            paragraphs_a[title]['answer_ids'], paragraphs_b[title]['answer_ids'] = answer_ids[:cutout], answer_ids[cutout:]
            # Eventual tokenized_questions split
            tokenized_questions = [paragraph['tokenized_questions'][i] for i in permutation]
            paragraphs_a[title]['tokenized_questions'], paragraphs_b[title]['tokenized_questions'] = tokenized_questions[:cutout], tokenized_questions[cutout:]
        return (
            TandaDataset.from_existing(paragraphs_a, pad_token_id=self.pad_token_id, sep_token_id=self.sep_token_id,),
            TandaDataset.from_existing(paragraphs_b, pad_token_id=self.pad_token_id, sep_token_id=self.sep_token_id,),
        )

    def split(
        self,
        p : float,
        seed : Optional[int] = None,
    ) -> Tuple[TandaDataset, TandaDataset]:
        """
        Splits the TandaDataset in two according to the proportion (p) passed.
        For reproducibility's sake, a (seed) can be passed too.
        """
        # For reproducibility
        self.shuffle(seed)
        # Accumulators
        paragraphs_a, paragraphs_b = {}, {}
        # Cutoff
        cutoff = int(p * len(self.yield_order))
        # Paragraph-level loop
        for i, (title, question_number) in enumerate(self.yield_order):
            # Paragraph accumulator choice
            paragraphs = paragraphs_a if i < cutoff else paragraphs_b
            # Check if the paragraph is already there, add the passages and tokenized stuff if it ain't
            if title not in paragraphs:
                paragraphs[title] = {
                    'questions' : [],
                    'answer_ids' : [],
                    'passages' : self.paragraphs[title]['passages'] + [],
                    'tokenized' : self.paragraphs[title]['tokenized'],
                    'tokenized_questions' : [],
                    'tokenized_passages' : self.paragraphs[title]['tokenized_passages'] if self.paragraphs[title]['tokenized'] else None,
                    }
            # Add the question and eventualy the tokenized question
            paragraphs[title]['questions'].append(self.paragraphs[title]['questions'][question_number])
            paragraphs[title]['answer_ids'].append(self.paragraphs[title]['answer_ids'][question_number])
            if paragraphs[title]['tokenized']:
                paragraphs[title]['tokenized_questions'].append(self.paragraphs[title]['tokenized_questions'][question_number])
        return (
            TandaDataset.from_existing(paragraphs_a, pad_token_id=self.pad_token_id, sep_token_id=self.sep_token_id,),
            TandaDataset.from_existing(paragraphs_b, pad_token_id=self.pad_token_id, sep_token_id=self.sep_token_id,),
        )

    def build_yield_order(self) -> None:
        """
        Builds the yield_order attribute. Check Dataset documentation for more information.
        """
        self.yield_order = []
        for title, paragraph in self.paragraphs.items():
            self.yield_order += [(title, i) for i in range(len(paragraph['questions']))]

    def tokenize(
        self,
        tokenizer: Tokenizer,
        **kwargs,
    ) -> None:
        """
        Tokenizes the questions and passages with the given (tokenizer).
        Tokenizer kwargs may be passed as (kwargs).
        """
        self.sep_token_id = tokenizer.sep_token_id
        self.pad_token_id = tokenizer.pad_token_id
        for _, paragraph in self.paragraphs.items():
            paragraph['tokenized'] = True
            paragraph['tokenized_questions'] = tokenizer(
                paragraph['questions'], **kwargs)['input_ids']
            paragraph['tokenized_passages'] = tokenizer(
                paragraph['passages'], **kwargs)['input_ids']

    def batches(
        self,
        force_balance : bool = False,
        batch_size: int = 1,
        max_length: int = 256,
        shuffle: bool = True,
    ) -> TandaBatch:
        """
        Yields the batches during training with a size of (batch_size).
        Also truncates the sequences to (max_length).
        """
        assert all([paragraph['tokenized'] for _, paragraph in self.paragraphs.items(
        )]), 'Some paragraphs are not tokenized, cannot yield batches.'
        if shuffle:
            self.shuffle()
        for i in range(0, len(self), batch_size):
            X, Y = [], []
            for title, question_nb in self.yield_order[i: i + batch_size]:
                question = self.paragraphs[title]['tokenized_questions'][question_nb]
                answer_id = self.paragraphs[title]['answer_ids'][question_nb]
                number_of_passages = len(self.paragraphs[title]['tokenized_passages'])
                if force_balance:
                    chosen = answer_id
                    if rd.random() < 0.5 or number_of_passages == 1: # Positive couple
                        pass
                    else: # Negative couple
                        while chosen == answer_id:
                            chosen = rd.randint(1, number_of_passages) - 1
                else:
                    chosen = rd.randint(1, number_of_passages) - 1
                answer = self.paragraphs[title]['tokenized_passages'][chosen]
                answer[0] = self.sep_token_id
                X.append(question + answer)
                Y.append(int(chosen == answer_id))
            X = pad_and_tensorize(X, self.pad_token_id, max_length, True)
            Y = torch.tensor(Y)
            yield TandaBatch(X, Y)

    def top_ranking_iterator(
        self,
        max_length : int = 256,
    ) -> TandaBatch:
        """
        Instead of using the normal yield order, yields the questions one by one with their associated passages.
        Limits the batch dimension 1 to (max_length).
        """
        for paragraph in self.paragraphs.values():
            for tokenized_question, answer_id in zip(paragraph['tokenized_questions'], paragraph['answer_ids']):
                X, Y = [], torch.tensor([answer_id])
                for tokenized_passage in paragraph['tokenized_passages']:
                    tokenized_passage[0] = self.sep_token_id
                    X.append(tokenized_question + tokenized_passage)
                X = pad_and_tensorize(X, self.pad_token_id, max_length, True)
                yield TandaBatch(X, Y)

    def __add__(self, other) -> TandaDataset:
        paragraphs = {}
        for title, paragraph in self.paragraphs.items():
            paragraphs[title] = paragraph
        for title, paragraph in other.paragraphs.items():
            while title in paragraphs:
                title = title + '_other'
            paragraphs[title] = paragraph
        sep_token_id = self.sep_token_id if self.sep_token_id == other.sep_token_id else None
        pad_token_id = self.pad_token_id if self.pad_token_id == other.pad_token_id else None
        return TandaDataset.from_existing(
            paragraphs=paragraphs,
            yield_order=None,
            sep_token_id=sep_token_id,
            pad_token_id=pad_token_id,
        )

    @staticmethod
    def from_existing(
        paragraphs : Dict[str, dict],
        yield_order : Optional[List[Tuple[str, int]]] = None,
        sep_token_id : Optional[int] = None,
        pad_token_id : Optional[int] = None,
    ) -> TandaDataset:
        tanda_dataset = TandaDataset({})
        tanda_dataset.paragraphs = paragraphs
        tanda_dataset.length = sum(len(paragraph['questions']) for _, paragraph in paragraphs.items())
        if yield_order is None:
            tanda_dataset.build_yield_order()
        else:
            tanda_dataset.yield_order = yield_order
        if sep_token_id is None or pad_token_id is None:
            for title in tanda_dataset.paragraphs:
                tanda_dataset.paragraphs[title]['tokenized'] = False
        else:
            tanda_dataset.sep_token_id = sep_token_id
            tanda_dataset.pad_token_id = pad_token_id
        return tanda_dataset

    @staticmethod
    def from_squad_like(
        squad_like : Union[Path, dict],
        ) -> TandaDataset:
        if isinstance(squad_like, Path):
            with open(squad_like, encoding='utf-8') as file:
                squad_like = json.load(file)['data']
        paragraphs, not_found = {}, 0
        for datum in squad_like:
            title = datum['title']
            for paragraph in datum['paragraphs']:
                paragraph_sentences = extract_sentences(paragraph)
                paragraph_questions, paragraph_answer_ids, paragraph_not_found = extract_qas(paragraph, paragraph_sentences, not_found)
                paragraphs[title] = (paragraph_questions, paragraph_sentences, paragraph_answer_ids)
                not_found += paragraph_not_found
        print(f"During data loading, {not_found} questions were not found.")
        return TandaDataset(paragraphs)
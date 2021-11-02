from typing import Union, List, Optional, Tuple

import torch
import re


Attention_mask = List[int]
Number = Union[int, float]



def pad_and_tensorize(
    list_of_lists : List[List[Number]],
    padding : Number = 0,
    max_length : Optional[int] = None, 
    return_attention_mask : bool = False,
) -> List[List[Number]]:
    """
    Pads a (list_of_list) with a given (padding) and returns it as a tensor.
    If a (max_length) argument is passed, the tensor dim=1 may be truncated.
    If the (return_attention_mask) is passed, also returns the attention masks. 
    """
    # Compute inner list max length
    inner_list_max_length = max(len(inner_list) for inner_list in list_of_lists)
    # If a max length is passed, ensures it's respected
    max_length = min(max_length, inner_list_max_length) if max_length is not None else inner_list_max_length
    # The tensor that's going to be returned
    tensor = torch.full(
        size=(len(list_of_lists), max_length),
        fill_value=padding,
    )
    # Eventual attention mask
    if return_attention_mask:
        attention_mask = torch.zeros(
            size=(len(list_of_lists), max_length),
            dtype=torch.int64,
        )
    # Inner lists loop
    for i, inner_list in enumerate(list_of_lists):
        # Compute inner list length
        inner_list_length = min(len(inner_list), max_length)
        # Place the inner list in the returned tensor
        tensor[i, :inner_list_length] = torch.tensor(inner_list[:inner_list_length])
        # Eventually place the inner list attention mask
        if return_attention_mask:
            attention_mask[i, :inner_list_length] = torch.ones(size=(inner_list_length,))
    # Returns
    if return_attention_mask:
        return tensor, attention_mask
    else:
        return tensor

# region from_squad_like related functions
def remove_abreviations(
    text : str,
) -> str:
    """
    Removes abbreviations from a (text).
    """
    text = re.sub('[A-Z]\.', lambda m: m.group(0)[0].lower(), text)
    text = text.replace('art.', 'article')
    text = text.replace('réf.', 'référence')
    text = text.replace('etc.', 'etc')
    text = text.replace('av.', 'avant')
    text = text.replace('av.', 'avant')
    text = text.replace('hab.', 'habitants')
    text = text.replace('Sr.', 'Senior')
    text = text.replace('°c', 'degrés Celsius')
    text = text.replace('°C', 'degrés Celsius')
    return text

def extract_sentences(
    paragraph : dict,
    min_length : int = 3,
) -> List[str]:
    """
    Extracts sentences from a (squad_like_datum) which length is superior to (min_length).
    """
    text_acc, already_added = '', set()
    text = remove_abreviations(paragraph['context'])
    if text not in already_added:
        text_acc = text_acc + ' ' + text
        already_added.add(text)
    sentences = []
    for sentence in text.split('.'):
        sentence = sentence.strip()
        if len(sentence) > min_length:
            sentences.append(sentence + '.')
    return sentences

def extract_qas(
    paragraph : dict,
    sentences : List[str],
    not_found : int,
) -> Tuple[List[str], List[int]]:
    questions, ids, not_found = [], [], 0
    for qas in paragraph['qas']:
        questions.append(qas['question'])
        answer = qas['answers'][0]['text']
        answer = remove_abreviations(answer).strip()
        for i, sentence in enumerate(sentences):
            if answer in sentence:
                ids.append(i)
                break
        if len(questions) != len(ids):
            not_found += 1
            questions.pop()
    return questions, ids, not_found
# endregion
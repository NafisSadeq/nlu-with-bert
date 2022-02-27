import logging
from typing import List, Union

import numpy as np
import torch
from intuit_topic.backend import BaseEmbedder
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch import Tensor
from tqdm import tqdm, trange


class SentenceTransformerWrapper(SentenceTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('CREATED SENTENCE TRANSFORMER WRAPPER')

    """
    Wrapper around Sentence Transformer to handle erroneous batches
    """
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        output_value: str = 'sentence_embedding',
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: str = None,
        normalize_embeddings: bool = False
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
                sentences, '__len__'
        ):  #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort(
            [-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0,
                                  len(sentences),
                                  batch_size,
                                  desc="Batches",
                                  disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index +
                                               batch_size]

            # Extract features!
            try:
                features = self.tokenize(sentences_batch)
            except:
                # Figure out which feature is bad and skip it
                old_batch = sentences_batch
                sentences_batch = []
                for s_ix, s in tqdm(enumerate(old_batch)):
                    try:
                        _ = self.tokenize([s])
                        sentences_batch.append(s)
                    except Exception as e:
                        print(e)
                        print(
                            'SKIPPING - Failed to tokenize sentence {} in batch: "{}"'
                            .format(s_ix, s))

                # If whole batch is bad, skip
                if not sentences_batch:
                    continue

                features = self.tokenize(sentences_batch)

            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(
                            out_features[output_value],
                            out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[
                                last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id + 1])
                elif output_value is None:  #Return all outputs
                    embeddings = []
                    for sent_idx in range(
                            len(out_features['sentence_embedding'])):
                        row = {
                            name: out_features[name][sent_idx]
                            for name in out_features
                        }
                        embeddings.append(row)
                else:  #Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings,
                                                                   p=2,
                                                                   dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [
            all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


class SentenceTransformerBackend(BaseEmbedder):
    """ Sentence-transformers embedding model

    The sentence-transformers embedding model used for generating document and
    word embeddings.

    Arguments:
        embedding_model: A sentence-transformers embedding model

    Usage:

    To create a model, you can load in a string pointing to a
    sentence-transformers model:

    ```python
    from intuit_topic.backend import SentenceTransformerBackend

    sentence_model = SentenceTransformerBackend("all-MiniLM-L6-v2")
    ```

    or  you can instantiate a model yourself:
    ```python
    from intuit_topic.backend import SentenceTransformerBackend
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    sentence_model = SentenceTransformerBackend(embedding_model)
    ```
    """
    def __init__(self,
                 embedding_model: Union[str, SentenceTransformer],
                 device: str = 'cpu'):
        super().__init__()

        if isinstance(embedding_model, SentenceTransformer):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = SentenceTransformerWrapper(embedding_model,
                                                              device=device)
        else:
            raise ValueError(
                "Please select a correct SentenceTransformers model: \n"
                "`from sentence_transformers import SentenceTransformer` \n"
                "`model = SentenceTransformer('all-MiniLM-L6-v2')`")

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings

        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process

        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        embeddings = self.embedding_model.encode(documents,
                                                 show_progress_bar=verbose)
        return embeddings

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains various classes and functions required for text generation with self-debiasing.
"""
from typing import List, Optional, Union, Tuple, Any

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, LogitsProcessorList, LogitsWarper, PreTrainedTokenizer, LogitsProcessor
from transformers import GPT2Tokenizer
from transformers.generation_utils import GenerationMixin, SampleOutput, SampleEncoderDecoderOutput, \
    SampleDecoderOnlyOutput, StoppingCriteriaList, validate_stopping_criteria


class SelfDebiasingLogitsProcessor(LogitsProcessor):
    """This class represents a logits processor that applies self-debiasing."""

    def __init__(self, num_debiasing_prefixes: int, decay_constant: float = 100, epsilon: float = 0.01,
                 tokenizer: Optional[PreTrainedTokenizer] = None, label: Any = None):
        """
        :param num_debiasing_prefixes: the number of debiasing prefixes used
        :param decay_constant: the decay constant (lambda in the paper)
        :param epsilon: the minimum factor by which each probability is multiplied
        """
        self.num_debiasing_prefixes = num_debiasing_prefixes
        self.decay_constant = decay_constant
        self.epsilon = epsilon
        self.tokenizer = tokenizer
        self.label = label

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        regular_batch_size = scores.shape[0] // (1 + self.num_debiasing_prefixes)
        for regular_sentence_idx in range(regular_batch_size):
            bias_indices = self._get_bias_indices(regular_sentence_idx, regular_batch_size)
            if bias_indices:
                self._debias_scores(scores, regular_sentence_idx, bias_indices)
        return scores

    def _get_bias_indices(self, regular_sentence_idx: int, batch_size: int) -> List[int]:
        """Returns the indices of all self-debiasing inputs for a regular input"""
        return [regular_sentence_idx + (prefix_idx + 1) * batch_size for prefix_idx in range(self.num_debiasing_prefixes)]

    def _debias_scores(self, scores: torch.FloatTensor, regular_sent_idx: int, bias_indices: List[int]) -> None:
        """Partially debiases the given scores considering a single sentence and the corresponding self-debiasing inputs"""
        logits_biased = [scores[bias_idx] for bias_idx in bias_indices]

        mask = self._generate_decay_mask(scores[regular_sent_idx], logits_biased)
        scores[regular_sent_idx] = torch.log(self._apply_decay_mask(scores[regular_sent_idx], mask))

        for debiasing_sent_idx in bias_indices:
            scores[debiasing_sent_idx] = scores[regular_sent_idx]

    def _apply_decay_mask(self, logits: torch.Tensor, decay_mask: torch.Tensor) -> torch.Tensor:
        """Applies exponential decay to a tensor of logits"""
        probabilities = logits.softmax(dim=-1)
        decay_mask = torch.exp(- decay_mask * self.decay_constant)
        decay_mask = torch.max(decay_mask, torch.tensor([self.epsilon], device=decay_mask.device))
        probabilities = probabilities * decay_mask
        probabilities = probabilities / probabilities.sum(dim=-1)
        return probabilities

    def _generate_decay_mask(self, logits_regular: torch.FloatTensor, logits_biased_list: List[torch.FloatTensor],
                             ) -> torch.Tensor:
        """Computes the alpha values (see paper) for each token and stores them in a mask tensor"""
        p_regular = logits_regular.softmax(dim=-1)
        p_biased = None

        for logits_biased in logits_biased_list:
            if p_biased is None:
                p_biased = logits_biased.softmax(dim=-1)
            else:
                p_biased = torch.max(p_biased, logits_biased.softmax(dim=-1))

        mask = torch.max(p_biased - p_regular, torch.tensor([0.], device=p_regular.device))
        return mask

    def _get_most_likely_tokens(self, probabilities_tensor: torch.Tensor, k: int) -> List[Tuple[str, float]]:
        """Returns the most likely tokens according to a tensor of probabilities"""
        assert len(probabilities_tensor.shape) == 1
        values, indices = torch.topk(probabilities_tensor, k=k, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(indices)
        return list(zip(tokens, [pv.item() for pv in values]))


class SelfDebiasingGPT2LMHeadModel(GPT2LMHeadModel, GenerationMixin):
    """
    This class represents a regular GPT2LMHeadModel that additionally has the capacity to perform self-debiasing. For self-debiasing, the
    init_logits_processor function must be called. Otherwise, this model just performs regular language modeling.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logits_processor = None

    def init_logits_processor(self, *args, **kwargs):
        self.logits_processor = SelfDebiasingLogitsProcessor(*args, **kwargs)

    def _get_logits_processor(self, *args, **kwargs) -> LogitsProcessorList:
        logits_processor = super()._get_logits_processor(*args, **kwargs)
        if self.logits_processor is not None:
            logits_processor.append(self.logits_processor)
        return logits_processor

    def sample(self, input_ids: torch.LongTensor, logits_processor: Optional[LogitsProcessorList] = None,
               logits_warper: Optional[LogitsProcessorList] = None, max_length: Optional[int] = None,
               stopping_criteria: Optional[StoppingCriteriaList] = None,
               pad_token_id: Optional[int] = None, eos_token_id: Optional[int] = None,
               output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
               output_scores: Optional[bool] = None, return_dict_in_generate: Optional[bool] = None,
               **model_kwargs) -> Union[
        SampleOutput, torch.LongTensor]:
        """
        This is a verbatim copy of the original implementation by huggingface, with a single modification to ensure that a text and all
        corresponding self-debiasing inputs always chose the same token to generate next. This modification is enclosed by the texts
        "BEGIN MODIFICATIONS" and "END MODIFICATIONS", respectively.
        """
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        # auto-regressive generation
        while True:

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # =========================
            # BEGIN MODIFICATIONS
            # the following modification to the sample method is necessary to ensure that each debiasing sentence is continued in the same
            # way as the original sentence
            if self.logits_processor is not None:
                regular_batch_size = next_tokens.shape[0] // (1 + self.logits_processor.num_debiasing_prefixes)
                for regular_sentence_idx in range(regular_batch_size):
                    debiasing_sentence_indices = self.logits_processor._get_bias_indices(regular_sentence_idx, regular_batch_size)
                    for debiasing_sentence_idx in debiasing_sentence_indices:
                        next_tokens[debiasing_sentence_idx] = next_tokens[regular_sentence_idx]
            # END MODIFICATIONS
            # =========================

            # add code that transfomers next_tokens to tokens_to_add
            if eos_token_id is not None:
                assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
                next_tokens = next_tokens * unfinished_sequences + (pad_token_id) * (1 - unfinished_sequences)

            # add token and increase length by one
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids


class GPT2Wrapper():

    def __init__(self, model_name: str = "gpt2-xl", use_cuda: bool = True):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-xl")
        :param use_cuda: whether to use CUDA
        """
        super().__init__()
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(model_name)
        if use_cuda:
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token  # original pad token id is 50257, not in embedding matrix
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

    def generate_self_debiasing(self, input_texts: List[str], debiasing_texts: List[str], num_samples: int = 1,
                                decay_constant: float = 0.0, min_length: int = None,
                                max_length: int = None, label: Any = None, **kwargs) -> List[str]:

        num_debiasing_prefixes = len(debiasing_texts) // len(input_texts)
        if decay_constant > 0:
            self._model.init_logits_processor(num_debiasing_prefixes=num_debiasing_prefixes,
                                              decay_constant=decay_constant,
                                              label=label, tokenizer=self._tokenizer)

        regular_batch_size = len(input_texts) * num_samples
        inputs = input_texts + debiasing_texts

        inputs = self._tokenizer(inputs, padding=True, return_tensors='pt')
        batch_size, seq_len = inputs['input_ids'].shape

        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = seq_len - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(batch_size):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        if min_length is not None:
            min_length = min_length + seq_len
        if max_length is not None:
            max_length = min(self._model.config.max_position_embeddings, max_length + seq_len)

        output_ids = self._model.generate(**inputs, min_length=min_length, max_length=max_length,
                                          num_return_sequences=num_samples, **kwargs)

        output_ids = output_ids[:regular_batch_size, seq_len:]
        return self._tokenizer.batch_decode(output_ids)

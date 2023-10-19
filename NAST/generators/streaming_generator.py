# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

import numpy as np
import torch
from fairseq import utils


DecoderOut = namedtuple(
    "StreamingDecoderOut",
    ["output_tokens", "output_scores", "attn", "step", "max_step", "history"],
)


class StreamingGenerator(object):
    def __init__(
        self,
        tgt_dict,
        models=None,
        retain_dropout=False,
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            retain_dropout: retaining dropout in the inference
        """
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.retain_dropout = retain_dropout
        self.models = models

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the StreamingGenerator is not supported"
            )

        if not self.retain_dropout:
            for model in models:
                model.eval()

        # TODO: streaming generator does not support ensemble for now.
        model = models[0]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        bsz, src_len = src_tokens.size()

        
        # initialize
        #encoder_out = model.forward_encoder([src_tokens, src_lengths])
        #prev_decoder_out = model.initialize_output_tokens(encoder_out, src_tokens)

        
        buffer = [[] for _ in range(bsz)]
        actions = [[] for _ in range(bsz)]
        num_read = 1
        
        while num_read < src_len + 1:
            
            partial_src_tokens = src_tokens[:, :num_read]
            partial_src_lengths = src_lengths.clamp(0, num_read)
            
            for i in range(bsz):
                if (src_tokens[i,num_read-1] != self.bos) and (src_tokens[i,num_read-1] != self.pad):
                    actions[i] += 'R'
            
            
            partial_encoder_out = model.forward_encoder([partial_src_tokens, partial_src_lengths])
            partial_prev_decoder_out = model.initialize_output_tokens(partial_encoder_out, partial_src_tokens)


            unpad_output_tokens = model.forward_streaming_decoder(
                partial_prev_decoder_out, partial_encoder_out, num_read
            )
            
            for i in range(bsz):
                if len(buffer[i]) == 0:
                    buffer[i] = unpad_output_tokens[i]
                    actions[i] += ['W' for i in range(len(unpad_output_tokens[i]))]
                elif len(unpad_output_tokens[i]) == 0:
                    continue
                else:
                    if buffer[i][-1] == unpad_output_tokens[i][0]:
                        buffer[i] += unpad_output_tokens[i][1:]
                        actions[i] += ['W' for i in range(len(unpad_output_tokens[i][1:]))]
                    else:
                        buffer[i] += unpad_output_tokens[i]
                        actions[i] += ['W' for i in range(len(unpad_output_tokens[i]))]

            num_read += 1

        return buffer, actions



# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import numpy as np
import logging
from typing import Union, Optional, Dict, List
from collections import Counter


import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATEncoder, FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from fairseq.modules.transformer_layer import TransformerDecoderLayerBase
from fairseq.distributed import fsdp_wrap

from .torch_imputer import best_alignment, imputer_loss
from ..generators.streaming_generator import DecoderOut


logger = logging.getLogger(__name__)

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    max_src_len = src_lens.max()
    bsz = src_lens.size(0)
    ratio = int(max_trg_len / max_src_len)
    index_t = utils.new_arange(trg_lens, max_src_len)
    index_t = torch.repeat_interleave(index_t, repeats=ratio, dim=-1).unsqueeze(0).expand(bsz, -1)
    return index_t 


@register_model("nonautoregressive_streaming_transformer")
class NATransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.plain_ctc = args.plain_ctc
        self.src_upsample_ratio = args.src_upsample_ratio

    @property
    def allow_length_beam(self):
        return True

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)

        # length prediction
        parser.add_argument(
            "--src-embedding-copy",
            action="store_true",
            help="copy encoder word embeddings as the initial input of the decoder",
        )
        parser.add_argument(
            "--src-upsample-ratio",
            type=int,
        )
        parser.add_argument(
            '--plain-ctc',
            action='store_true',
        )
        parser.add_argument(
            '--ctc-beam-size',
            type=int
        )
        parser.add_argument(
            '--wait-until',
            type=int
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        encoder = UniTransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            encoder.apply(init_bert_params)
        return encoder
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def sequence_ngram_loss_with_logits(self, logits, logit_mask, targets):
        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        
        if not self.args.sctc_loss:
            if self.args.ngram_size == 1:
                loss = self.compute_ctc_1gram_loss(log_probs, logit_mask, targets)
            elif self.args.ngram_size == 2:
                loss = self.compute_ctc_bigram_loss(log_probs, logit_mask, targets)
            else:
                raise NotImplementedError
        else:
            loss = self.compute_sctc_ngram_loss(log_probs, targets, self.args.ngram_size)
        
        return loss
    
    
    def compute_ctc_1gram_loss(self, log_probs, logit_mask, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        bow = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        bow[:,self.tgt_dict.blank_index] = 0
        ref_bow = torch.zeros(batch_size, vocab_size).cuda(probs.get_device())
        ones = torch.ones(batch_size, vocab_size).cuda(probs.get_device())
        ref_bow.scatter_add_(-1, targets, ones).detach()
        ref_bow[:,self.pad] = 0
        expected_length = torch.sum(bow).div(batch_size)
        loss = torch.mean(torch.norm(bow-ref_bow,p=1,dim=-1))/ (length_tgt + expected_length)
        return loss

    def compute_ctc_bigram_loss(self, log_probs, logit_mask, targets):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        targets = targets.tolist()
        probs_blank = probs[:,:,self.tgt_dict.blank_index]
        length = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        length[:,self.tgt_dict.blank_index] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,self.tgt_dict.blank_index]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_1 = []
        gram_2 = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - 1
        for i in range(batch_size):
            two_grams = Counter()
            gram_1.append([])
            gram_2.append([])
            gram_count.append([])
            for j in range(num_grams):
                two_grams[(targets[i][j], targets[i][j+1])] += 1
            j = 0
            for two_gram in two_grams:
                if self.pad in two_gram:
                    continue
                gram_1[-1].append(two_gram[0])
                gram_2[-1].append(two_gram[1])
                gram_count[-1].append(two_grams[two_gram])
                if two_gram[0] == two_gram[1]:
                    rep_gram_pos.append((i, j))
                j += 1
            while len(gram_count[-1]) < num_grams:
                gram_1[-1].append(1)
                gram_2[-1].append(1)
                gram_count[-1].append(0)
        gram_1 = torch.LongTensor(gram_1).cuda(blank_matrix.get_device())
        gram_2 = torch.LongTensor(gram_2).cuda(blank_matrix.get_device())
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        gram_1_probs = torch.gather(probs, -1, gram_1.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, length_ctc, 1)
        gram_2_probs = torch.gather(probs, -1, gram_2.view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
        probs_matrix = torch.matmul(gram_1_probs, gram_2_probs)
        bag_grams = blank_matrix.view(batch_size, 1, length_ctc, length_ctc) * probs_matrix
        bag_grams = torch.sum(bag_grams.view(batch_size, num_grams, -1), dim = -1).view(batch_size, num_grams,1)
        if len(rep_gram_pos) > 0:
            for pos in rep_gram_pos:
                i, j = pos
                gram_id = gram_1[i, j]
                gram_prob = probs[i, :, gram_id]
                rep_gram_prob = torch.sum(gram_prob[1:] * gram_prob[:-1])
                bag_grams[i, j, 0] = bag_grams[i, j, 0] - rep_gram_prob
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)


        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2)
        
        return loss

    def compute_sctc_ngram_loss(self, log_probs, logit_mask, targets, n):

        batch_size, length_ctc, vocab_size = log_probs.size()
        _, length_tgt = targets.size()
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)
        targets = targets.tolist()
        probs_blank = probs[:,:,self.tgt_dict.blank_index]
        length = probs[:,0,] + torch.sum(probs[:,1:,:] * (1 - probs[:,:-1,:]), dim = 1)
        length[:,self.tgt_dict.blank_index] = 0
        expected_length = torch.sum(length).div(batch_size)

        logprobs_blank = log_probs[:,:,self.tgt_dict.blank_index]
        cumsum_blank = torch.cumsum(logprobs_blank, dim = -1)
        cumsum_blank_A = cumsum_blank.view(batch_size, 1, length_ctc).expand(-1, length_ctc, -1)
        cumsum_blank_B = cumsum_blank.view(batch_size, length_ctc, 1).expand(-1, -1, length_ctc)
        cumsum_blank_sub = cumsum_blank_A - cumsum_blank_B
        cumsum_blank_sub = torch.cat((torch.zeros(batch_size, length_ctc,1).cuda(cumsum_blank_sub.get_device()), cumsum_blank_sub[:,:,:-1]), dim = -1)
        tri_mask = torch.tril(utils.fill_with_neg_inf(torch.zeros([batch_size, length_ctc, length_ctc]).cuda(cumsum_blank_sub.get_device())), 0)
        cumsum_blank_sub = cumsum_blank_sub + tri_mask
        blank_matrix = torch.exp(cumsum_blank_sub)

        gram_idx = []
        gram_count = []
        rep_gram_pos = []
        num_grams = length_tgt - n + 1
        for i in range(batch_size):
            ngrams = Counter()
            gram_idx.append([])
            gram_count.append([])
            for j in range(num_grams):
                idx = []
                for k in range(n):
                    idx.append(targets[i][j+k])
                idx = tuple(idx)
                ngrams[idx] += 1

            for k in range(n):
                gram_idx[-1].append([])
            for ngram in ngrams:
                for k in range(n):
                    gram_idx[-1][k].append(ngram[k])
                gram_count[-1].append(ngrams[ngram])

            while len(gram_count[-1]) < num_grams:
                for k in range(n):
                    gram_idx[-1][k].append(1)
                gram_count[-1].append(0)

        gram_idx = torch.LongTensor(gram_idx).cuda(blank_matrix.get_device()).transpose(0,1)
        gram_count = torch.Tensor(gram_count).cuda(blank_matrix.get_device()).view(batch_size, num_grams,1)
        blank_matrix = blank_matrix.view(batch_size, 1, length_ctc, length_ctc)
        for k in range(n):
            gram_k_probs = torch.gather(probs, -1, gram_idx[k].view(batch_size,1,num_grams).expand(batch_size,length_ctc,num_grams)).transpose(1,2).view(batch_size, num_grams, 1, length_ctc)
            if k == 0:
                state = gram_k_probs
            else:
                state = torch.matmul(state, blank_matrix) * gram_k_probs
        bag_grams = torch.sum(state, dim=-1)
        match_gram = torch.min(torch.cat([bag_grams,gram_count],dim = -1), dim = -1)[0]
        match_gram = torch.sum(match_gram).div(batch_size)
        
        assert match_gram <= length_tgt - (n-1)
        assert match_gram <= expected_length - (n-1)
        
        loss = (- 2 * match_gram).div(length_tgt + expected_length - 2*(n - 1))
        return loss


    def sequence_ctc_loss_with_logits(self,
                                      logits: torch.FloatTensor,
                                      logit_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      targets: torch.LongTensor,
                                      target_mask: Union[torch.FloatTensor, torch.BoolTensor],
                                      blank_index: torch.LongTensor,
                                      label_smoothing=0,
                                      reduce=True
                                      ) -> torch.FloatTensor:
        # lengths : (batch_size, )
        # calculated by counting number of mask
        logit_lengths = (logit_mask.bool()).long().sum(1)

        if len(targets.size()) == 1:
            targets = targets.unsqueeze(0)
            target_mask = target_mask.unsqueeze(0)
        target_lengths = (target_mask.bool()).long().sum(1)

        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        # log_probs_T : (T, batch_size, n_class), this kind of shape is required for ctc_loss
        log_probs_T = log_probs.transpose(0, 1)
        #     assert (target_lengths == 0).any()
        targets = targets.long()
        targets = targets[target_mask.bool()]
        if reduce:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="mean",
                zero_infinity=True,
            )
        else:
            loss = F.ctc_loss(
                log_probs_T.float(),  # compatible with fp16
                targets,
                logit_lengths,
                target_lengths,
                blank=blank_index,
                reduction="none",
                zero_infinity=True,
            )
            loss = torch.stack([a / b for a, b in zip(loss, target_lengths)])

        n_invalid_samples = (logit_lengths < target_lengths).long().sum()

        if n_invalid_samples > 0:
            logger.warning(
                f"The length of predicted alignment is shoter than target length, increase upsample factor: {n_invalid_samples} samples"
            )
            # raise ValueError

        if label_smoothing > 0:
            smoothed_loss = -log_probs.mean(-1)[logit_mask.bool()].mean()
            loss = (1 - label_smoothing) * loss + label_smoothing * smoothed_loss
        return loss
    

    def _compute_latency_loss(self, logits, logit_mask, source_mask):
        
        # (batch_size, T, n_class)
        log_probs = logits.log_softmax(-1)
        src_lengths = source_mask.sum(dim=-1) 
        
        batch_size, length_ctc, vocab_size = log_probs.size()
        
        probs = torch.exp(log_probs)
        probs = probs.masked_fill(~logit_mask.unsqueeze(-1), 0)

        s = self.src_upsample_ratio
        
        cut_src_lengths = src_lengths.unsqueeze(-1) - 1
        cut_source_mask = torch.scatter(source_mask, -1, cut_src_lengths, 0) #src=torch.tensor([False]).unsqueeze(-1).to(source_mask))
        cut_logit_mask = cut_source_mask.unsqueeze(-1).expand(batch_size, -1, s).reshape(batch_size, -1)
        
        cut_probs = probs.masked_fill(~cut_logit_mask.unsqueeze(-1), 0)
        
        suffix_probs = cut_probs[:,1:,:] * (1 - cut_probs[:,:-1,:])
        cut_length = torch.cat([cut_probs[:,0,:].unsqueeze(1),suffix_probs],dim=1)    
        cut_length[:,:,self.tgt_dict.blank_index] = 0
        cut_length[:,:,self.tgt_dict.bos()] = 0
        cut_emit_prob = cut_length.sum(dim=-1)
        expected_cut_length = cut_emit_prob.sum(dim=-1)
        

        latency_vtx = torch.arange(0, int(length_ctc/s)).to(cut_emit_prob)
        latency_vtx = torch.repeat_interleave(latency_vtx, repeats=s).unsqueeze(0)
        tol_latency = (cut_emit_prob * latency_vtx).sum(dim=-1)
        al_approx = (tol_latency - 0.5 * src_lengths * (expected_cut_length - 1)) / expected_cut_length
        al_approx = torch.clamp(al_approx, min=self.args.latency_threshold)
        return al_approx.mean()
    
    
    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, glat, reduce=True, **kwargs
    ):
        prev_output_tokens = self.initialize_output_tokens_by_upsampling(src_tokens)
        
        prev_output_tokens_mask = prev_output_tokens.ne(self.pad)
        output_length = prev_output_tokens_mask.sum(dim=-1)
        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)

        target_mask = tgt_tokens.ne(self.pad)
        target_length = target_mask.sum(dim=-1) 
        # glat_implemented_here
        glat_info = None
        oracle = None
        keep_word_mask = None
        
        if glat and tgt_tokens is not None:
            with torch.set_grad_enabled(glat.get('require_glance_grad', False)):
                normalized_logits = self.decoder(
                    normalize=True,
                    prev_output_tokens=prev_output_tokens,
                    encoder_out=encoder_out,
                )

                normalized_logits_T = normalized_logits.transpose(0, 1).float() #T * B * C, float for FP16

                best_aligns = best_alignment(normalized_logits_T, tgt_tokens, output_length, target_length, self.tgt_dict.blank_index, zero_infinity=True)
                #pad those positions with <blank>
                padded_best_aligns = torch.tensor([a + [0] * (normalized_logits_T.size(0) - len(a)) for a in best_aligns], device=prev_output_tokens.device, dtype=prev_output_tokens.dtype)
                oracle_pos = (padded_best_aligns // 2).clip(max=tgt_tokens.size(-1)-1)
                oracle = tgt_tokens.gather(-1, oracle_pos)
                oracle = oracle.masked_fill(padded_best_aligns % 2 == 0, self.tgt_dict.blank_index)
                oracle = oracle.masked_fill(~prev_output_tokens_mask, self.pad)
                
                _,pred_tokens = normalized_logits.max(-1)
                same_num = ((pred_tokens == oracle) & prev_output_tokens_mask).sum(dim=-1)
                keep_prob = ((output_length - same_num) / output_length * glat['context_p']).unsqueeze(-1) * prev_output_tokens_mask.float()

                keep_word_mask = (torch.rand(prev_output_tokens.shape, device=prev_output_tokens.device) < keep_prob).bool()
        
                glat_prev_output_tokens = prev_output_tokens.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)

                glat_info = {
                    "glat_acc": (same_num.sum() / output_length.sum()).detach(),
                    "glat_context_p": glat['context_p'],
                    "glat_keep": keep_prob.mean().detach(),
                }
                prev_output_tokens = glat_prev_output_tokens                  
        

        # decoding
        word_ins_out = self.decoder(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )

        target_mask = tgt_tokens.ne(self.pad)
        
        if self.args.use_ngram:
            ctc_loss = self.sequence_ngram_loss_with_logits(word_ins_out, prev_output_tokens_mask, tgt_tokens)
        else:    
            ctc_loss = self.sequence_ctc_loss_with_logits(
                logits=word_ins_out,
                logit_mask=prev_output_tokens_mask,
                targets=tgt_tokens,
                target_mask=target_mask,
                blank_index=self.tgt_dict.blank_index,
                label_smoothing=self.args.label_smoothing,
                reduce=reduce
            )
        
        source_mask = src_tokens.ne(self.pad)
        if self.args.latency_factor != 0 :
            latency_loss = self._compute_latency_loss(word_ins_out, prev_output_tokens_mask, source_mask)
        else:
            latency_loss = torch.Tensor([0]).to(ctc_loss)
        
        ret_val = {
            "ctc_loss": {"loss": ctc_loss},
            "latency_loss": {"loss": latency_loss, "factor": self.args.latency_factor},
        }
        return ret_val, glat_info

    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):        
        step = decoder_out.step
        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_lengths = (output_masks.bool()).long().sum(-1) 
        output_logits = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=step,
        )
 
        _scores, _tokens = output_logits.max(-1)
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])
        output_scores.masked_scatter_(output_masks, _scores[output_masks])
        if history is not None:
            history.append(output_tokens.clone())

        def _ctc_postprocess(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
            hyp = [v for v in deduplicated_toks if (v != self.tgt_dict.blank_index) and (v!= self.tgt_dict.pad_index)]
            return hyp
        
        
        unpad_output_tokens = []
        for output_token in output_tokens:
            unpad_output_tokens.append(_ctc_postprocess(output_token))

        res_lengths = torch.tensor([len(res) for res in unpad_output_tokens],device=decoder_out.output_tokens.device, dtype=torch.long)
        res_seqlen = max(res_lengths.tolist())
        res_tokens = [res + [self.tgt_dict.pad_index] * (res_seqlen - len(res)) for res in unpad_output_tokens]
        res_tokens = torch.tensor(res_tokens, device=decoder_out.output_tokens.device, dtype=decoder_out.output_tokens.dtype)


        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None,
            history=history,
        )

            
            
            
    def forward_streaming_decoder(self, decoder_out, encoder_out, num_generate, **kwargs):        

        output_tokens = decoder_out.output_tokens

        # execute the decoder
        output_masks = output_tokens.ne(self.pad)
        output_logits = self.decoder(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
            step=0,
        )
        
        if not self.plain_ctc:
            return output_logits[:,(num_generate-1) * self.src_upsample_ratio : num_generate * self.src_upsample_ratio,:]
   
        _, _tokens = output_logits.max(-1)
        output_tokens = output_tokens.clone()
        output_tokens.masked_scatter_(output_masks, _tokens[output_masks])

        def _ctc_postprocess(tokens):
            _toks = tokens.int().tolist()
            deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
            hyp = [v for v in deduplicated_toks if (v != self.tgt_dict.blank_index) and (v!= self.tgt_dict.pad_index) and (v!= self.tgt_dict.bos())]
            return hyp
        
        
        unpad_output_tokens = []
        for output_token in output_tokens:
            unpad_output_tokens.append(_ctc_postprocess(output_token[(num_generate-1) * self.src_upsample_ratio : num_generate * self.src_upsample_ratio]))


        return unpad_output_tokens

    def initialize_output_tokens_by_upsampling(self, src_tokens):
        if self.src_upsample_ratio <= 1:
            return src_tokens

        def _us(x, s):
            B = x.size(0)
            _x = x.unsqueeze(-1).expand(B, -1, s).reshape(B, -1)
            return _x

        return _us(src_tokens, self.src_upsample_ratio)
        
    
    def initialize_output_tokens(self, encoder_out, src_tokens):
        initial_output_tokens = self.initialize_output_tokens_by_upsampling(src_tokens)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

    def regenerate_length_beam(self, decoder_out, beam_size):
        output_tokens = decoder_out.output_tokens
        length_tgt = output_tokens.ne(self.pad).sum(1)
        length_tgt = (
            length_tgt[:, None]
            + utils.new_arange(length_tgt, 1, beam_size)
            - beam_size // 2
        )
        length_tgt = length_tgt.view(-1).clamp_(min=2)
        max_length = length_tgt.max()
        idx_length = utils.new_arange(length_tgt, max_length)

        initial_output_tokens = output_tokens.new_zeros(
            length_tgt.size(0), max_length
        ).fill_(self.pad)
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk
        )
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos)

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(decoder_out.output_scores)

        return decoder_out._replace(
            output_tokens=initial_output_tokens, output_scores=initial_output_scores
        )

class UniTransformerEncoder(FairseqNATEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)


    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        #construct unidirectional attention mask for encoder 
        dim_t = x.size(0)
        init_uni_mask = torch.ones([dim_t, dim_t], device=x.device, dtype=x.dtype)
        uni_mask = torch.triu(init_uni_mask, diagonal=1)
        
        
        # encoder layers
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=uni_mask
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }
        

class NATransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()

        self.encoder_embed_dim = args.encoder_embed_dim
        self.src_embedding_copy = getattr(args, "src_embedding_copy", False)
        self.embed_length = Embedding(256, self.encoder_embed_dim, None)
        self.src_upsample_ratio = args.src_upsample_ratio
        self.wait_until = args.wait_until

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = ModifiedTransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    
    @ensemble_decoder
    def forward(self, normalize, encoder_out, prev_output_tokens, step=0, oracle=None, keep_word_mask=None, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            embedding_copy=(step == 0) & self.src_embedding_copy,
            oracle=oracle,
            keep_word_mask=keep_word_mask,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out


    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        embedding_copy=False,
        oracle=None,
        keep_word_mask=None,
        **unused
    ):
        """
        Similar to *forward* but only return features.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        if embedding_copy:
            src_embd = encoder_out["encoder_embedding"][0]
            if len(encoder_out["encoder_padding_mask"]) > 0:
                src_mask = encoder_out["encoder_padding_mask"][0]
            else:
                src_mask = None
            src_mask = (
                ~src_mask
                if src_mask is not None
                else prev_output_tokens.new_ones(*src_embd.size()[:2]).bool()
            )

            x, decoder_padding_mask = self.forward_embedding(
                prev_output_tokens,
                self.forward_copying_source(
                    src_embd, src_mask, prev_output_tokens.ne(self.padding_idx)
                ),
            )
            if oracle is not None:
                oracle_embedding, _ = self.forward_embedding(oracle)
                x = x.masked_fill(keep_word_mask.unsqueeze(-1), 0) + oracle_embedding.masked_fill(~keep_word_mask.unsqueeze(-1), 0)
                

        else:

            x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        dim_t_src = encoder_out["encoder_out"][0].size(0)
        
        cross_attn_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim_t_src, dim_t_src])), 1 + self.wait_until
            ).to(x)
        cross_attn_mask = torch.repeat_interleave(cross_attn_mask, repeats=self.src_upsample_ratio, dim=0)
        
        block_mask = torch.ones([self.src_upsample_ratio,self.src_upsample_ratio], dtype=torch.bool, device=x.device)
        block_mask_list = [block_mask for i in range(dim_t_src)]
        block_diag_mask = torch.block_diag(*block_mask_list)
        self_attn_mask = block_diag_mask + torch.tril(torch.ones_like(block_diag_mask), 0)
        self_attn_mask = utils.fill_with_neg_inf(torch.zeros([self_attn_mask.size(0), self_attn_mask.size(0)], device=x.device)).masked_fill(self_attn_mask, 0).to(x)
        #self_attn_mask[:(self.wait_until + 1)*self.src_upsample_ratio,:(self.wait_until + 1)*self.src_upsample_ratio] = 0
        
        # decoder layers
        for i, layer in enumerate(self.layers):

            # early exit from the decoder.
            if (early_exit is not None) and (i >= early_exit):
                break

            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=decoder_padding_mask,
                cross_attn_mask=cross_attn_mask,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    def forward_embedding(self, prev_output_tokens, states=None):
        # embed positions
        positions = (
            self.embed_positions(prev_output_tokens)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)
            if self.project_in_dim is not None:
                x = self.project_in_dim(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        return x, decoder_padding_mask

    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding


class ModifiedTransformerDecoderLayerBase(TransformerDecoderLayerBase):
    '''
    modify the forward function to add the ''cross_attn_mask'' argument
    '''
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_mask=cross_attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None
    


@register_model_architecture(
    "nonautoregressive_streaming_transformer", "nonautoregressive_streaming_transformer"
)
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)
    args.src_upsample_ratio = getattr(args, "src_upsample_ratio", 2)
    args.plain_ctc = getattr(args, "plain_ctc", False)
    args.ctc_beam_size = getattr(args, "ctc_beam_size", 20)
    args.wait_until = getattr(args, "wait_until", 0)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (AdaptiveSoftmax, SinusoidalPositionalEmbedding)
from . import (
    FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model,
    register_model_architecture,
)

import ipdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
def _bp_hook_factory(n):
    def _bp_hook(grad):
        if np.isinf(grad.detach().cpu().numpy()).any():
            import pdb; pdb.set_trace()
        if torch.isnan(grad).any():
            print('{} grad is NaN!'.format(n))
    return _bp_hook

def _isnan(x):
    return np.isnan(x.detach().cpu().numpy()).any()
def _isinf(x):
    return np.isinf(x.detach().cpu().numpy()).any()

@register_model('abdnmt')
class ABDNMT(FairseqModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-hidden-size', type=int, metavar='N',
                            help='decoder hidden size')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='number of decoder layers')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--share-decoder-input-output-embed',
                            action='store_false',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')

        parser.add_argument('--common-dropout', type=int)
        parser.add_argument('--rnn-dropout', type=int)
        parser.add_argument('--head-count', type=int)
        parser.add_argument('--src-pe', action='store_true')
        parser.add_argument('--tgt-pe', action='store_true')
        parser.add_argument('--only-bd', action='store_true')
        parser.add_argument('--bd-write', action='store_true')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        num_embeddings = len(task.source_dictionary)
        pretrained_encoder_embed = Embedding(
            num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
        )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise ValueError('--share-all-embeddings requires a joint dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise ValueError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        encoder = Encoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            common_dropout=args.common_dropout,
            rnn_dropout=args.rnn_dropout,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
            src_pe=args.src_pe,
            max_source_positions=args.max_source_positions,
        )
        decoder = ForwardDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            common_dropout=args.common_dropout,
            rnn_dropout=args.rnn_dropout,
            head_count=args.head_count,
            encoder_output_units=encoder.output_units,
            share_input_output_embed=args.share_decoder_input_output_embed,
            pretrained_embed=pretrained_decoder_embed,
            tgt_pe=args.tgt_pe,
            max_target_positions=args.max_target_positions,
            only_bd=args.only_bd,
            bd_write=args.bd_write,
        )
        return cls(encoder, decoder)


class Encoder(FairseqEncoder):
    """RNMT encoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=2,
        common_dropout=0.5, rnn_dropout=0.3, bidirectional=True,
        left_pad=True, padding_value=0., pretrained_embed=None,
        src_pe=False, max_source_positions=-1,
    ):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.common_dropout = common_dropout
        self.rnn_dropout = rnn_dropout
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.gru = GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.rnn_dropout if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

        if src_pe:
            self.src_pe = SinusoidalPositionalEmbedding(embed_dim, self.padding_idx, False, max_source_positions + self.padding_idx + 1)
            self.embed_scale = math.sqrt(embed_dim)
        else:
            self.src_pe = None

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )
        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        if self.src_pe:
            x = self.src_pe(src_tokens) * self.embed_scale + x
        x = F.dropout(x, p=self.common_dropout, training=self.training, inplace=True)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        packed_outs, _ = self.gru(packed_x, h0)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        #x = F.dropout(x, p=self.common_dropout, training=self.training, inplace=True)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        return {
            'encoder_out': x,
            'encoder_padding_mask': encoder_padding_mask
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


class MultiHeadMLPAttention(nn.Module):
    def __init__(self, key_dim, query_dim, model_dim, head_count, dim_per_head=None):

        super(MultiHeadMLPAttention, self).__init__()

        if dim_per_head is None:
            assert model_dim % head_count == 0
            dim_per_head = model_dim // head_count

        self.head_count = head_count

        self.dim_per_head = dim_per_head

        self.model_dim = model_dim

        self.linear_keys = Linear(key_dim, head_count * self.dim_per_head)
        self.linear_query = Linear(query_dim, head_count * self.dim_per_head)
        self.linear_weight = Linear(self.dim_per_head, 1)
        self.final_linear = Linear(key_dim, key_dim)
        self.sm = nn.Softmax(dim=-1)

    def _split_heads(self, x):

        batch_size = x.size(0)

        return x.view(batch_size, -1, self.head_count, x.size(-1) // self.head_count).transpose(1, 2).contiguous()

    def _combine_heads(self, x):

        seq_len = x.size(2)

        return x.transpose(1, 2).contiguous().view(-1, seq_len, self.head_count * x.size(-1))

    def forward(self, key, value, query, mask=None, enc_attn_cache=None):
        batch_size = key.size(0)

        # 1) Project key, value, and query.
        if enc_attn_cache is not None:
            key_up, value_up = enc_attn_cache
        else:
            key_up = self._split_heads(self.linear_keys(key))  # [batch_size, num_head, seq_len, dim_head]
            value_up = self._split_heads(value)

        query_up = self._split_heads(self.linear_query(query))

        key_len = key_up.size(2)
        query_len = query_up.size(2)

        # 2) Calculate and scale scores.
        scores = self.linear_weight(torch.tanh(query_up + key_up)).squeeze(-1)

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, float('-inf'))

        # 3) Apply attention dropout and compute context vectors.
        attn = self.sm(scores).unsqueeze(2)
        context = self._combine_heads(torch.matmul(attn, value_up))

        output = self.final_linear(context)

        # END CHECK
        return output, attn, [key_up, value_up]


class BackwardDecoder(FairseqIncrementalDecoder):
    """ABDNMT backward decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, common_dropout=0.5, rnn_dropout=0.3, head_count=8,
        encoder_output_units=1024, share_input_output_embed=True, pretrained_embed=None,
        tgt_pe=False, left_pad=False, max_target_positions=-1,
    ):
        super().__init__(dictionary)
        self.common_dropout = common_dropout
        self.rnn_dropout = rnn_dropout
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        self.gru1 = GRUCell(embed_dim, hidden_size)
        self.gru2 = GRUCell(encoder_output_units, hidden_size)
        self.linear_bridge = Linear(encoder_output_units // 2, hidden_size)
        self.attention = MultiHeadMLPAttention(encoder_output_units, hidden_size, hidden_size, head_count)
        self.linear_output = Linear(embed_dim + hidden_size + encoder_output_units, out_embed_dim)
        if not self.share_input_output_embed:
            self.linear_proj = Linear(out_embed_dim, num_embeddings)

        if tgt_pe:
            self.tgt_pe = SinusoidalPositionalEmbedding(embed_dim, padding_idx, left_pad, max_target_positions + padding_idx + 1)
            self.embed_scale = math.sqrt(embed_dim)
        else:
            self.tgt_pe = None
        self.eos_idx = dictionary.eos()
        self.padding_idx = padding_idx

    def forward(self, prev_output_tokens, encoder_out_dict, test=False):
        """
        use with forward decoder
        :param prev_output_tokens:
        :param encoder_out_dict:
        :return:
        """
        tgt_lengths = prev_output_tokens.ne(self.padding_idx).sum(-1)
        if not test:
            prev_output_tokens = self.r2l(prev_output_tokens, tgt_lengths)

        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        bsz, seqlen = prev_output_tokens.size()

        srclen = encoder_out.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.common_dropout, training=self.training, inplace=True)

        # B x T x C -> T x B x C
        emb = x.transpose(0, 1)

        no_padding_mask = 1 - encoder_padding_mask
        # B X 1
        encoder_lengths = no_padding_mask.sum(0, keepdim=True).t()
        # see https://github.com/pytorch/pytorch/issues/3587
        # see https://pytorch.org/docs/stable/nn.html#gru
        # B x S x 2C
        enc_states = encoder_out.transpose(0, 1)
        # B x 1 x 2C
        enc_forward_idx = (encoder_lengths - 1).unsqueeze(-1).expand(-1, -1, enc_states.size(-1))
        forward_n = torch.gather(enc_states, 1, enc_forward_idx)
        forward_n = forward_n.view(bsz, 1, 2, self.encoder_output_units // 2)[:, 0, 0, :]
        prev_hiddens = torch.tanh(self.linear_bridge(forward_n))
        attn_cache = None

        init_hidden = prev_hiddens

        if not test:
            # teacher forcing
            hiddens = []
            attn_ctxs = []
            for j in range(seqlen):
                tmp_hidden = self.gru1(emb[j, :, :], prev_hiddens)
                attn_ctx, _, attn_cache = self.attention(key=encoder_out.transpose(0, 1),
                                                         value=encoder_out.transpose(0, 1),
                                                         query=tmp_hidden.unsqueeze(1),
                                                         mask=encoder_padding_mask.transpose(0, 1),
                                                         enc_attn_cache=attn_cache)
                attn_ctx = attn_ctx.squeeze(1)
                prev_hiddens = self.gru2(attn_ctx, tmp_hidden)
                hiddens.append(prev_hiddens)
                attn_ctxs.append(attn_ctx)

            # B x T x H
            hiddens = torch.stack(hiddens, 1)
            attn_ctxs = torch.stack(attn_ctxs, 1)
            logits = F.dropout(
                torch.tanh(
                    self.linear_output(
                        torch.cat((emb.transpose(0, 1), hiddens, attn_ctxs), -1)
                    )
                ),
                p=self.common_dropout,
                training=self.training,
                inplace=False
            )
            del hiddens, attn_ctxs
            if self.share_input_output_embed:
                output = F.linear(logits, self.embed_tokens.weight)
            else:
                output = self.linear_proj(logits)
            #del logits
        else:
            output = None
            logits = None

        # greedy search
        gs_emb = emb[0]  # first token, B x C
        gs_states = []
        gs_best_logits = []
        gs_states_mask = []
        gs_has_eos = encoder_padding_mask.new_zeros(emb.size(1))  # B
        gs_hidden = init_hidden
        gs_attn_cache = attn_cache
        max_len = srclen * 2
        counter = 0
        while counter < max_len:
            gs_tmp_hidden = self.gru1(gs_emb, gs_hidden)
            gs_attn_ctx, _, gs_attn_cache = self.attention(key=encoder_out.transpose(0, 1),
                                               value=encoder_out.transpose(0, 1),
                                               query=gs_tmp_hidden.unsqueeze(1),
                                               mask=encoder_padding_mask.transpose(0, 1),
                                               enc_attn_cache=gs_attn_cache)
            gs_attn_ctx = gs_attn_ctx.squeeze(1)
            gs_hidden = self.gru2(gs_attn_ctx, gs_tmp_hidden)

            gs_logit = F.dropout(
                torch.tanh(
                    self.linear_output(
                        torch.cat((gs_emb, gs_hidden, gs_attn_ctx), -1)
                    )
                ),
                p=self.common_dropout,
                training=self.training,
                inplace=True,
            )
            del gs_attn_ctx

            if self.share_input_output_embed:
                gs_logit = F.linear(gs_logit, self.embed_tokens.weight)
            else:
                gs_logit = self.linear_proj(gs_logit)
            gs_best_logit = torch.max(F.log_softmax(gs_logit, dim=-1), -1)[1]  # B
            gs_best_logits.append(gs_best_logit)
            del gs_logit

            # use previous step gs_has_eos to calculate mask
            gs_hidden[gs_has_eos.nonzero()] = 0
            gs_states.append(gs_hidden)
            # 0 means token, 1 means pad
            gs_mask_t = torch.zeros_like(gs_has_eos)
            gs_mask_t[gs_has_eos.nonzero()] = 1
            gs_states_mask.append(gs_mask_t)

            gs_has_eos[torch.eq(gs_best_logit, self.eos_idx).nonzero()] = 1
            if gs_has_eos.nonzero().size(0) == bsz:
                break
            gs_emb = self.embed_tokens(gs_best_logit)
            #gs_emb = F.dropout(gs_emb, p=self.common_dropout, training=self.training)
            counter += 1
        gs_states_mask = torch.stack(gs_states_mask, 1)  # [batch_size, infer_len]
        gs_states = torch.stack(gs_states, 1)  # [batch_size, infer_len, hidden_size]
        gs_best_logits = torch.stack(gs_best_logits, 1)
        # INFO("infer length -> {} ratio -> {}".format(counter, counter / seq_len))

        #gs_states.register_hook(_bp_hook_factory('gs_states'))
        #output.register_hook(_bp_hook_factory('bd_output_l2r'))
        if not test:
            output = self.l2r(output, tgt_lengths)
            logits = self.l2r(logits, tgt_lengths)
        #output.register_hook(_bp_hook_factory('bd_output_r2l'))

        return output, logits, gs_states, gs_states_mask, gs_best_logits

    def only_bd_forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        """use for only_bd option"""
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        positions = self.tgt_pe(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.tgt_pe is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        tgt_lengths = prev_output_tokens.ne(self.padding_idx).sum(-1)
        if self.training or (not self.training and incremental_state is None):
            prev_output_tokens = self.r2l(prev_output_tokens, tgt_lengths)

        bsz, seqlen = prev_output_tokens.size()

        srclen = encoder_out.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x = positions * self.embed_scale + x
        x = F.dropout(x, p=self.common_dropout, training=self.training, inplace=False)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # all B x S x H inside attention
        if cached_state is not None:
            prev_hiddens, attn_cache = cached_state
        else:
            no_padding_mask = 1 - encoder_padding_mask
            # B X 1
            encoder_lengths = no_padding_mask.sum(0, keepdim=True).t()
            # see https://github.com/pytorch/pytorch/issues/3587
            # see https://pytorch.org/docs/stable/nn.html#gru
            # B x S x 2C
            enc_states = encoder_out.transpose(0, 1)
            # B x 1 x 2C
            enc_forward_idx = (encoder_lengths - 1).unsqueeze(-1).expand(-1, -1, enc_states.size(-1))
            forward_n = torch.gather(enc_states, 1, enc_forward_idx)
            forward_n = forward_n.view(bsz, 1, 2, self.encoder_output_units // 2)[:, 0, 0, :]
            prev_hiddens = torch.tanh(self.linear_bridge(forward_n))
            attn_cache = None

        hiddens = []
        attn_ctxs = []
        for j in range(seqlen):
            tmp_hidden = self.gru1(x[j, :, :], prev_hiddens)
            attn_ctx, attn_weight, attn_cache = self.attention(key=encoder_out.transpose(0, 1),
                                                               value=encoder_out.transpose(0, 1),
                                                               query=tmp_hidden.unsqueeze(1),
                                                               mask=encoder_padding_mask.transpose(0, 1),
                                                               enc_attn_cache=attn_cache)
            attn_ctx = attn_ctx.squeeze(1)
            prev_hiddens = self.gru2(attn_ctx, tmp_hidden)
            hiddens.append(prev_hiddens)
            attn_ctxs.append(attn_ctx)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, attn_cache),
        )

        # B x T x H
        hiddens = torch.stack(hiddens, 1)
        attn_ctxs = torch.stack(attn_ctxs, 1)
        logits = F.dropout(
            torch.tanh(
                self.linear_output(
                    torch.cat((x.transpose(0, 1), hiddens, attn_ctxs), -1)
                )
            ),
            p=self.common_dropout,
            training=self.training,
            inplace=False,
        )
        if self.share_input_output_embed:
            output = F.linear(logits, self.embed_tokens.weight)
        else:
            output = self.linear_proj(logits)

        if self.training or (not self.training and incremental_state is None):
            output = self.l2r(output, tgt_lengths)

        return output, None

    def r2l(self, x, lengths):
        if x.size(1) == 1:
            return x
        reorder_idx = [
            torch.cat([
                x.new_zeros(1),
                torch.arange(lengths[i] - 1, 0, -1, device=x.device),
                torch.arange(lengths[i], x.size(1), 1, device=x.device)
            ])
            for i in range(x.size(0))
        ]
        reorder_idx = torch.stack(reorder_idx, 0)
        x = torch.gather(x, 1, reorder_idx)
        return x

    def l2r(self, x, lengths):
        if x.size(1) == 1:
            return x
        reorder_idx = [
            torch.cat([
                torch.arange(lengths[i] - 2, -1, -1, device=x.device),
                torch.arange(lengths[i] - 1, x.size(1), 1, device=x.device)
            ])
            for i in range(x.size(0))
        ]
        reorder_idx = torch.stack(reorder_idx, 0).unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.gather(x, 1, reorder_idx)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class ForwardDecoder(FairseqIncrementalDecoder):
    """ABDNMT forward decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, common_dropout=0.5, rnn_dropout=0.3, head_count=8,
        encoder_output_units=1024, share_input_output_embed=True, pretrained_embed=None,
        tgt_pe=False, left_pad=False, max_target_positions=-1, only_bd=False, bd_write=True,
    ):
        super().__init__(dictionary)
        self.common_dropout = common_dropout
        self.rnn_dropout = rnn_dropout
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.padding_idx=padding_idx
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        self.linear_bridge = Linear(encoder_output_units // 2, hidden_size)
        self.gru1 = GRUCell(embed_dim, hidden_size)
        self.gru2 = GRUCell(encoder_output_units + hidden_size, hidden_size)
        self.attention = MultiHeadMLPAttention(encoder_output_units, hidden_size, hidden_size, head_count)
        self.bd_attention = MultiHeadMLPAttention(hidden_size, hidden_size, hidden_size, head_count)
        self.bd_forget = Linear(hidden_size, hidden_size)
        self.bd_update = Linear(hidden_size, hidden_size)
        self.linear_output = Linear(embed_dim + hidden_size * 2 + encoder_output_units, out_embed_dim)
        if not self.share_input_output_embed:
            self.linear_proj = Linear(out_embed_dim, num_embeddings)

        if tgt_pe:
            self.tgt_pe = SinusoidalPositionalEmbedding(embed_dim, padding_idx, left_pad, max_target_positions + padding_idx + 1)
            self.embed_scale = math.sqrt(embed_dim)
        else:
            self.tgt_pe = None

        self.backward_decoder = BackwardDecoder(dictionary, embed_dim=embed_dim, hidden_size=hidden_size,
                                                out_embed_dim=out_embed_dim,
                                                num_layers=num_layers, common_dropout=common_dropout,
                                                rnn_dropout=rnn_dropout, head_count=head_count,
                                                encoder_output_units=encoder_output_units,
                                                share_input_output_embed=share_input_output_embed,
                                                pretrained_embed=self.embed_tokens)
        self.only_bd = only_bd
        self.bd_write = bd_write
        self.tgt_dict = dictionary
        from tensorboardX import SummaryWriter
        self.tb = SummaryWriter(log_dir=f'bw_{bd_write}_new_font')
        self.cnt = 0

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        if self.only_bd:
            return self.backward_decoder.only_bd_forward(prev_output_tokens, encoder_out_dict, incremental_state)
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        positions = self.tgt_pe(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.tgt_pe is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        srclen = encoder_out.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        if positions is not None:
            x = positions * self.embed_scale + x
        x = F.dropout(x, p=self.common_dropout, training=self.training, inplace=True)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        # all B x S x H inside attention
        bd_output = None
        bd_logits = None
        bd_gs_best_logits = None
        if cached_state is not None:
            prev_hiddens, src_attn_cache, bd_attn_cache, bd_states, bd_states_mask = cached_state
        else:
            # see https://github.com/pytorch/pytorch/issues/3587
            # see https://pytorch.org/docs/stable/nn.html#gru
            backward_1 = encoder_out.transpose(0, 1).view(bsz, srclen, 2, self.encoder_output_units // 2)[:, 0, 1, :]
            prev_hiddens = torch.tanh(self.linear_bridge(backward_1))
            src_attn_cache = None
            bd_output, bd_logits, bd_states, bd_states_mask, bd_gs_best_logits = self.backward_decoder(prev_output_tokens, encoder_out_dict, test=(not
                self.training and incremental_state is not None))
            bd_attn_cache = None

        hiddens = []
        attn_ctxs = []
        align=[]
        for j in range(seqlen):
            tmp_hidden = self.gru1(x[j, :, :], prev_hiddens)
            src_attn_ctx, _, src_attn_cache = self.attention(key=encoder_out.transpose(0, 1),
                                                             value=encoder_out.transpose(0, 1),
                                                             query=tmp_hidden.unsqueeze(1),
                                                             mask=encoder_padding_mask.transpose(0, 1),
                                                             enc_attn_cache=src_attn_cache)
            bd_attn_ctx, bd_attn_weight, bd_attn_cache = self.bd_attention(key=bd_states,
                                                                           value=bd_states,
                                                                           query=tmp_hidden.unsqueeze(1),
                                                                           mask=bd_states_mask,
                                                                           enc_attn_cache=bd_attn_cache)
            align.append(bd_attn_weight.view(-1, bd_attn_weight.size(-1)))
            attn_ctx = torch.cat([src_attn_ctx.squeeze(1), bd_attn_ctx.squeeze(1)], -1)
            prev_hiddens = self.gru2(attn_ctx, tmp_hidden)
            if self.bd_write:
                # (batch_size, hidden_size)
                bd_forget_gate = torch.sigmoid(self.bd_forget(prev_hiddens))
                bd_update_gate = torch.sigmoid(self.bd_update(prev_hiddens))
                # (batch_size, head, 1, key_len) -> # (batch_size, key_len, head, 1)
                bd_attn_weight_1 = bd_attn_weight.permute(0, 3, 1, 2).contiguous()
                # (batch_size, head, key_len, dim_per_head) -> (batch_size, key_len, head, dim_per_head)
                new_bd_states = bd_attn_cache[1].permute(0, 2, 1, 3).contiguous()
                # -> (batch_size, key_len, head, dim_per_head)
                bd_attn_weight_1 = bd_attn_weight_1.expand(-1, -1, -1, new_bd_states.size(-1)).contiguous()
                # -> (batch_size, key_len, hidden_size)
                new_bd_states = new_bd_states.view(*(new_bd_states.size()[:2]), -1)
                # -> (batch_size, key_len, hidden_size)
                bd_attn_weight_1 = bd_attn_weight_1.view(*(bd_attn_weight_1.size()[:2]), -1)
                new_bd_states = new_bd_states * (
                1 - bd_attn_weight_1 * bd_forget_gate.unsqueeze(1)) + bd_attn_weight_1 * bd_update_gate.unsqueeze(1)
                # 1 means pad, 0 means token
                bd_states_mask_1 = bd_states_mask.unsqueeze(-1).float()
                if 'Half' in bd_states.type():
                    bd_states_mask_1 = bd_states_mask_1.half()
                bd_states = new_bd_states * (1 - bd_states_mask_1)
                bd_attn_cache = None
            hiddens.append(prev_hiddens)
            attn_ctxs.append(attn_ctx)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, src_attn_cache, bd_attn_cache, bd_states, bd_states_mask),
        )
        align=torch.stack(align, 1).cpu().detach().numpy()
        if not self.training:
            self.cnt += 1
            tgt_mask=prev_output_tokens.ne(self.padding_idx).long()
            tgt_length=tgt_mask.sum(-1).repeat_interleave(8,dim=0).cpu().detach().numpy()
            bd_length=(1-bd_states_mask).sum(-1).repeat_interleave(8,dim=0).cpu().detach().numpy()
            align=align*np.expand_dims(tgt_mask.cpu().detach().numpy().repeat(8,axis=0), -1)
            align=align[:,:,::-1]
            row=prev_output_tokens.cpu().detach().numpy().flatten()
            row=np.array(list(map(lambda x: self.tgt_dict[x], row))).reshape(prev_output_tokens.size())
            col=bd_gs_best_logits.cpu().detach().numpy().flatten()
            col=np.array(list(map(lambda x: self.tgt_dict[x], col))).reshape(bd_gs_best_logits.size())
            col=col[:,::-1]
            plt.switch_backend('agg')
            figs={}
            for i in range(align.shape[0]):
                if not (self.cnt == 1 and i == 101) and not (self.cnt == 1 and i == 98):
                    continue
                if tgt_length[i] < 15 or bd_length[i] < 0.5*tgt_length[i]:
                    continue
                tmp_arr=align[i, :tgt_length[i], -bd_length[i]:]
                upper=np.triu(tmp_arr).sum()
                tmp_arr_sum=tmp_arr.sum()
                if self.bd_write:
                    if upper < (0.9*tmp_arr_sum) or np.trace(tmp_arr) < 0.2*tmp_arr_sum:
                        continue
                if not self.bd_write:
                    if upper > 0.9*tmp_arr_sum or np.trace(tmp_arr) > 0.9*tmp_arr_sum:
                        continue
                #if self.bd_write is not True and not (self.cnt == 7 and i//8 ==32) and not (self.cnt == 2 and i//8==24):
                #    continue
                fig, ax=plt.subplots()
                im, cbar=heatmap(align[i,:tgt_length[i],-bd_length[i]:],row[i//8,:tgt_length[i]],col[i//8,-bd_length[i]:],ax=ax,cmap='gray')
                #texts=annotate_heatmap(im, valfmt='{x:.1f}')
                fig.tight_layout()
                self.tb.add_figure(f'{self.bd_write}/{self.cnt}_{i}',fig)

        # B x T x H
        hiddens = torch.stack(hiddens, 1)
        attn_ctxs = torch.stack(attn_ctxs, 1)
        #ipdb.set_trace()
        logits = F.dropout(
            torch.tanh(
                self.linear_output(
                    torch.cat((x.transpose(0, 1), hiddens, attn_ctxs), -1)
                )
            ),
            p=self.common_dropout,
            training=self.training,
            inplace=False
        )
        #logits.register_hook(_bp_hook_factory('logits'))
        if self.share_input_output_embed:
            output = F.linear(logits, self.embed_tokens.weight)
        else:
            output = self.linear_proj(logits)
        #output.register_hook(_bp_hook_factory('fd output'))
        return output, {'bd_output': bd_output, 'logits': logits, 'bd_logits': bd_logits}

    def reorder_incremental_state(self, incremental_state, new_order):
        if self.only_bd:
            self.backward_decoder.reorder_incremental_state(incremental_state, new_order)
            return
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if state is None:
                return state
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out
        #ipdb.set_trace()

        if self.only_bd:
            logits = net_output[0]
            if log_probs:
                return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
            else:
                return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        logits = net_output[0]
        bd_logits = net_output[1]['bd_output']
        if log_probs:
            if bd_logits is not None:
                return (utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace),
                    utils.log_softmax(bd_logits, dim=-1, onnx_trace=self.onnx_trace))
            else:
                return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace), None
        else:
            if bd_logits is not None:
                return (utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace),
                    utils.softmax(bd_logits, dim=-1, onnx_trace=self.onnx_trace))
            else:
                return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace), None


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.08, 0.08)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def GRU(input_size, hidden_size, **kwargs):
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.08, 0.08)
    return m


def GRUCell(input_size, hidden_size, **kwargs):
    m = nn.GRUCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.08, 0.08)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.08, 0.08)
    if bias:
        m.bias.data.uniform_(-0.08, 0.08)
    return m


@register_model_architecture('abdnmt', 'abdnmt_base_nist')
def base_architecture(args):
    args.common_dropout = getattr(args, 'common_dropout', 0.5)
    args.rnn_dropout = getattr(args, 'rnn_dropout', 0.3)
    args.head_count = getattr(args, 'head_count', 8)
    args.label_smoothing = getattr(args, 'label_smoothing', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 512)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.src_pe = getattr(args, 'src_pe', False)
    args.tgt_pe = getattr(args, 'tgt_pe', False)
    args.only_bd = getattr(args, 'only_bd', False)
    args.bd_write = getattr(args, 'bd_write', False)

@register_model_architecture('abdnmt', 'abdnmt_base_ende')
def base_architecture(args):
    args.common_dropout = getattr(args, 'common_dropout', 0.3)
    args.rnn_dropout = getattr(args, 'rnn_dropout', 0.1)
    args.head_count = getattr(args, 'head_count', 8)
    args.label_smoothing = getattr(args, 'label_smoothing', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', args.decoder_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 1024)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', True)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', True)
    args.src_pe = getattr(args, 'src_pe', False)
    args.tgt_pe = getattr(args, 'tgt_pe', False)
    args.only_bd = getattr(args, 'only_bd', False)
    args.bd_write = getattr(args, 'bd_write', False)

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    fd={'fontsize': 12, 'fontweight': 'black'}
    ax.set_xticklabels(col_labels,fontdict=fd)
    ax.set_yticklabels(row_labels,fontdict=fd)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

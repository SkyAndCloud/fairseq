# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import numpy as np
import torch
import ipdb
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

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.fd_weight = args.fd_weight
        self.bd_weight = args.bd_weight
        self.agree_weight = args.agree_weight
        self.only_bd = args.only_bd

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--fd-weight', default=1., type=float)
        parser.add_argument('--bd-weight', default=1., type=float)
        parser.add_argument('--agree-weight', default=1., type=float)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        #import ipdb; ipdb.set_trace()
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        if self.only_bd:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
        else:
            lprobs, bd_lprobs = model.get_normalized_probs(net_output, log_probs=True)
        #ipdb.set_trace()
        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)
        # fd loss
        lprobs = lprobs.view(-1, lprobs.size(-1))
        #lprobs.register_hook(_bp_hook_factory('fd lprobs'))
        nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
        if reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss
        #loss.register_hook(_bp_hook_factory('fd loss'))
        if not self.only_bd and self.training:
            # bd loss
            bd_lprobs = bd_lprobs.view(-1, bd_lprobs.size(-1))
            bd_nll_loss = -bd_lprobs.gather(dim=-1, index=target)[non_pad_mask]
            bd_smooth_loss = -bd_lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]
            if reduce:
                bd_nll_loss = bd_nll_loss.sum()
                bd_smooth_loss = bd_smooth_loss.sum()
            bd_eps_i = self.eps / bd_lprobs.size(-1)
            bd_loss = (1. - self.eps) * bd_nll_loss + bd_eps_i * bd_smooth_loss
            # agree loss
            bd_logits = net_output[1]['bd_logits']
            bd_logits = bd_logits.view(-1, bd_logits.size(-1))
            logits = net_output[1]['logits']
            logits = logits.view(-1, logits.size(-1))
            agree_loss = torch.sqrt((bd_logits - logits).pow(2).sum(dim=-1) + 1e-8)
            agree_loss = torch.sum(agree_loss * non_pad_mask.squeeze(-1).float())
            #agree_loss=0
            # total loss
            total_loss = self.fd_weight * loss + self.bd_weight * bd_loss + self.agree_weight * agree_loss
            total_nll_loss = self.fd_weight * nll_loss + self.bd_weight * bd_nll_loss
        else:
            total_loss = loss
            total_nll_loss = nll_loss

        #total_loss.register_hook(_bp_hook_factory('total_loss'))

        return total_loss, total_nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

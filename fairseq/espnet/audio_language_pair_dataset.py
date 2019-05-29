# -*- coding: utf-8 -*-
# author: 王树根
# email: wangshugen@ict.ac.cn
# date: 2019-05-28 15:34
import numpy as np
import torch

from fairseq.data import LanguagePairDataset
from fairseq.espnet.io_utils import LoadInputsAndTargets
from fairseq.espnet.nets_utils import pad_list


class CustomConverter(object):
    """Custom batch converter for Pytorch

    :param int subsampling_factor : The subsampling factor
    """

    def __init__(self, subsampling_factor=1, preprocess_conf=None):
        self.subsampling_factor = subsampling_factor
        self.load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=True, preprocess_conf=preprocess_conf)
        self.ignore_id = -1

    def transform(self, item):
        return self.load_inputs_and_targets(item)

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device

        :param list batch: The batch to transform
        :param torch.device device: The device to send to
        :return: a tuple xs_pad, ilens, ys_pad
        :rtype (torch.Tensor, torch.Tensor, torch.Tensor)
        """
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


class AudioLanguagePairDataset(LanguagePairDataset):

    def __init__(
            self, src, src_sizes, src_dict, audio, converter,
            tgt=None, tgt_sizes=None, tgt_dict=None,
            left_pad_source=True, left_pad_target=False,
            max_source_positions=1024, max_target_positions=1024,
            shuffle=True, input_feeding=True,
            remove_eos_from_source=False, append_eos_to_target=False,
    ):
        super().__init__(
            src, src_sizes, src_dict, tgt=tgt, tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict, left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            max_source_positions=max_source_positions,
            max_target_positions=max_target_positions, shuffle=shuffle,
            input_feeding=input_feeding,
            remove_eos_from_source=remove_eos_from_source,
            append_eos_to_target=append_eos_to_target
        )

        self.audio = audio
        assert isinstance(converter, CustomConverter)
        self.converter = converter

    def __getitem__(self, item):
        # sample is a dict containing id, source and target
        sample = super(self).__getitem__(item)
        sample.update({
            'audio': self.audio_dataset[item]
        })
        return sample

    def collater(self, samples):
        samples = super().collater(samples)
        for sample  in samples:
            sid = sample['id']
            ai = self.audio[sid]
            out = self.converter.transform(ai)
            sample['audio'] = out
        return samples

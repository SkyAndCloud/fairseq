#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: 王树根
# email: wangshugen@ict.ac.cn
# date: 2019-06-13 16:47
import contextlib
import os
import re
import warnings
from argparse import ArgumentParser
from collections import OrderedDict
from functools import partial
from os.path import abspath, basename

from easy_tornado.utils.file_operation import (
    create_if_not_exists,
    concat_path,
    write_file_contents,
    write_iterable_as_lines
)
from easy_tornado.utils.file_operation import load_file_contents
from easy_tornado.utils.str_extension import from_json, to_json


@contextlib.contextmanager
def none_context():
    yield


@contextlib.contextmanager
def work_dir(path=None):
    cwd = os.getcwd()
    try:
        if path is not None:
            os.chdir(path)
            yield
    finally:
        os.chdir(cwd)


def expand_lines(results):
    lines = []
    for rs in results:
        for line in rs[-1]:
            lines.append(line)
    return lines


def main():
    parser = ArgumentParser(description='For CCMT2019 test set')
    parser.add_argument(
        '--asr-rec-path', '-i', required=True,
        help='ASR recognition result path'
    )
    parser.add_argument(
        '--meta-file-name', '-mfn', default='mpp.json',
        help='meta data path'
    )
    parser.add_argument(
        '--max-length', '-ml', type=int, default=None,
        help='output path'
    )
    parser.add_argument(
        '--n-best', type=int, default=None,
        help='combined to one file'
    )
    parser.add_argument(
        '--output', '-o', default='./output',
        help='output path'
    )
    warnings.warn('--no-split no longer have any effect, always in this mode')
    args, _ = parser.parse_known_args()
    input_path = args.asr_rec_path
    output_path = abspath(args.output)
    create_if_not_exists(output_path)

    meta_data = OrderedDict({'offset': 0, 'lines': [], 'keys': []})
    with work_dir(input_path):
        files = os.listdir('.')
        for file in sorted(files):
            _meta_data = OrderedDict({'offset': meta_data['offset']})
            key = basename(file)
            meta_data[key] = _meta_data
            lines = extract(
                file, _meta_data, args.max_length, args.n_best
            )
            meta_data['lines'].extend(lines)
            meta_data['keys'].append(key)

    file_path = concat_path(output_path, 'source.txt')
    write_iterable_as_lines(
        file_path, expand_lines(meta_data['lines'])
    )

    meta_path = concat_path(output_path, args.meta_file_name)
    write_file_contents(meta_path, to_json(meta_data, indent=2))


def _do_newline(m, num, before=False):
    _matched = m.group(0)
    if len(_matched) != num:
        return _matched
    if _matched == '.':
        return _matched
    _sym = '\n' * num
    if before:
        return '{}{}'.format(_sym, _matched)
    else:
        return '{}{}'.format(_matched, _sym)


def newline(m, num=1):
    return _do_newline(m, num)


def newline_before(m, num=1):
    return _do_newline(m, num, before=True)


def comma_split(line):
    COMMA = u'，'
    new_line = re.sub(COMMA, newline, line)
    for p in new_line.split('\n'):
        yield p


def modal_split(line):
    _newline = partial(newline, num=2)
    MODAL_PATTERN_A = u'(吧。|啊！|啦！|吗？|呐。)'
    line = re.sub(MODAL_PATTERN_A, _newline, line)

    _newline = partial(newline_before, num=2)
    MODAL_PATTERN_B = u'(如果|假如|虽然|但是|因为|所以|尽管)'
    line = re.sub(MODAL_PATTERN_B, _newline, line)

    for p in line.split('\n\n'):
        yield p


def zh_split(line):
    ZH_PATTERN = u'[！|。|...|？|\?|!|：|；|~|～]'
    new_line = re.sub(ZH_PATTERN, newline, line)
    for s in new_line.split('\n'):
        yield s


def force_split(subject, max_length):
    sentences = []
    zh_gen = zh_split(subject)
    for sentence in zh_gen:
        # 空句子
        if sentence == '' or sentence.strip() == '':
            continue

        if len(sentence) <= max_length:
            sentences.append(sentence)
            continue

        # 句子过长，按照逗号拆分
        _cur_len = 0
        _tmp_sentence = ''
        for part in comma_split(sentence):
            _len = len(part)
            if _cur_len > 0:
                if _cur_len <= max_length:
                    if _cur_len + _len > max_length:
                        sentences.append(_tmp_sentence)
                        _cur_len = _len
                        _tmp_sentence = part
                    else:
                        _cur_len += _len
                        _tmp_sentence += part
                else:
                    sentences.append(_tmp_sentence)
                    _cur_len = _len
                    _tmp_sentence = part
            else:
                _cur_len = _len
                _tmp_sentence = part
        if _cur_len > 0:
            sentences.append(_tmp_sentence)
    return sentences


def process_samples(samples, max_length=None):
    filtered = [
        (i, x.strip()) for i, x in enumerate(samples)
        if x.strip() != ''
    ]

    rs = []
    for i, s in filtered:
        if max_length is None or len(s) <= max_length:
            rs.append((i, [0], [s]))
        else:
            indices, lines = [], []
            rs.append((i, indices, lines))
            for j, p in enumerate(force_split(s, max_length)):
                indices.append(j)
                lines.append(p)
    return rs


def extract(file, meta_data, max_length, n_best):
    assert n_best is None or n_best > 0
    records = load_file_contents(file)
    lines = []
    for i, record in enumerate(records, start=1):
        data = from_json(record)
        results = data['results_recognition']
        results2 = data['origin_result']['result']['word']
        assert results == results2
        results = process_samples(results, max_length)

        results = results[:n_best]
        assert len(results) > 0
        meta_data[i] = {
            'offset': meta_data['offset'],
            'indices': [x[:2] for x in results],
            'sn_start_time': data['sn_start_time'],
            'sn_end_time': data['sn_end_time']
        }
        meta_data['offset'] += len(results)
        lines.extend(results)
    return lines


if __name__ == '__main__':
    main()

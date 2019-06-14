#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: 王树根
# email: wangshugen@ict.ac.cn
# date: 2019-06-13 16:48
from argparse import ArgumentParser
from os.path import abspath

from easy_tornado.utils.file_operation import (
    create_if_not_exists,
    concat_path,
    load_file_contents,
    file_exists)
from easy_tornado.utils.str_extension import from_json, to_json


def main():
    parser = ArgumentParser(description='For CCMT2019 test set')
    parser.add_argument(
        '--asr-trans-path', '-i', required=True,
        help='translation result path'
    )
    parser.add_argument(
        '--meta-file-name', '-mfn', default='mpp.json',
        help='meta data path'
    )
    parser.add_argument(
        '--output', '-o', default='./output',
        help='output path'
    )
    args, _ = parser.parse_known_args()
    input_path = args.asr_trans_path
    output_path = abspath(args.output)
    create_if_not_exists(output_path)

    meta_path = concat_path(input_path, args.meta_file_name)
    meta_data = from_json(
        load_file_contents(meta_path, pieces=False)
    )

    offset = meta_data['offset']
    assert offset == 0
    file_path = concat_path(input_path, 'target.txt')
    assert file_exists(file_path)
    rfp = open(file_path, mode='r')
    for key in meta_data['keys']:
        file_path = concat_path(
            output_path, key.replace('.txt', '.json')
        )
        wfp = open(file_path, mode='w')
        _rs = []
        records = []

        _meta_data = meta_data[key]
        _offset = _meta_data.pop('offset')
        for i in sorted([int(x) for x in _meta_data.keys()]):
            k = str(i)
            rs = {}
            for index in _meta_data[k]['indices']:
                splits = [rfp.readline().strip() for _ in index[1]]
                s = {
                    "translation": ' '.join(splits),
                    "sn_start_time": _meta_data[k]['sn_start_time'],
                    "sn_end_time": _meta_data[k]['sn_end_time']
                }
                wfp.write('{}\n'.format(to_json(s)))
                _rs.extend(splits)
                offset += len(splits)
                rs[index[0]] = splits
            records.append(rs)
        wfp.close()
        print('{}: {}'.format(key, len(records)))


if __name__ == '__main__':
    main()

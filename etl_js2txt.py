#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: 王树根
# email: wangshugen@ict.ac.cn
# date: 2019-06-19 09:32
from argparse import ArgumentParser

from easy_tornado.utils.file_operation import (
    file_exists,
    load_file_contents,
    write_iterable_as_lines
)
from easy_tornado.utils.str_extension import from_json, to_json


def main():
    parser = ArgumentParser(description='For CCMT2019 test submit set')
    parser.add_argument(
        '--input-submit-jsons', '-i', nargs='+', required=True,
        help='files of submission format'
    )
    parser.add_argument(
        '--reverse', '-r', action='store_true', default=False,
        help='reverse converting'
    )
    args, _ = parser.parse_known_args()
    input_submit_jsons = args.input_submit_jsons
    for input_submit_json in input_submit_jsons:
        if not file_exists(input_submit_json):
            raise FileExistsError(input_submit_json)

        txt_path = input_submit_json.replace('.json', '.txt')
        json_lines = load_file_contents(input_submit_json)

        results = []
        if args.reverse:
            texts = load_file_contents(txt_path)
            assert len(json_lines) == len(texts)
        else:
            texts = []

        for i, json_line in enumerate(json_lines):
            data = from_json(json_line.strip())
            if args.reverse:
                data['translation'] = texts[i]
                results.append(to_json(data))
            else:
                results.append(data['translation'])

        if args.reverse:
            write_iterable_as_lines(input_submit_json, results)
        else:
            write_iterable_as_lines(txt_path, results)


if __name__ == '__main__':
    main()

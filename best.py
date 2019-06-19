#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: 王树根
# email: wangshugen@ict.ac.cn
# date: 2019-06-19 11:11
from math import inf

from easy_tornado.utils.file_operation import load_file_contents, write_iterable_as_lines

id_path = 'lm_test/res.id'
sent_path = 'lm_test/res.out'
score_path = 'lm_test/res.score'

id_contents = load_file_contents(id_path)
sent_contents = load_file_contents(sent_path)
score_contents = load_file_contents(score_path)

assert len(id_contents) == len(sent_contents) == len(score_contents)
ids, targets, last_id, best_score, best_sent = set(), [], None, -inf, None
for _id, sent, score in zip(id_contents, sent_contents, score_contents):
    score = float(score)
    ids.add(_id)
    if last_id is None or last_id != _id:
        if last_id is not None:
            print(last_id)
            targets.append(best_sent)
        last_id = _id
        best_score = score
        best_sent = sent
    else:
        if score > best_score:
            best_score = score
            best_sent = sent
targets.append(best_sent)
assert len(ids) == len(targets)
write_iterable_as_lines('./lm_test/res.best', targets)

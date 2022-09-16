# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from time import time


def Tictoc():
    """
    装饰器：时间计算
    tic用于计算起始时间，toc用于输出当前时段时间
    :return:
    """

    start_stack = []
    start_named = {}

    def tic(name=None):
        if name is None:
            start_stack.append(time())
        else:
            start_named[name] = time()

    def toc(name=None):
        if name is None:
            start = start_stack.pop()
        else:
            start = start_named.pop(name)
        elapsed = time() - start
        return elapsed

    return tic, toc

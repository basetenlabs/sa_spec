# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

import os


DISABLE_SA_SPEC = os.environ.get("DISABLE_SA_SPEC", "0") == "1"
if DISABLE_SA_SPEC:
    def add_request(*args, **kwargs):
        pass
    
    def prepare(*args, **kwargs):
        pass
    
    def extend(*args, **kwargs):
        pass
    
    def get_active_tokens_for_test(*args, **kwargs):
        pass

    SA_SPEC_THRESH = float("inf")


else:
    from ._torch_adapter import *
    from ._sa_spec_impl import *


    SA_SPEC_THRESH = int(os.environ.get("SA_SPEC_THRESH", "8"))

__all__ = [
    "add_request",
    "prepare",
    "extend",
    "SA_SPEC_THRESH",
]

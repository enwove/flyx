# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Reference implementation used to produce golden answers for flyx."""
from .draft import Draft, DraftConfig, DraftState
from .dspine import DSpine, DSpineConfig
from .dv import Dv, DvConfig, LlamaConfig
from .eagle4 import Eagle4, Eagle4Config
from .generate import greedy_plain, greedy_spec, greedy_spec_eagle4, greedy_spec_mtp
from .mtp import Mtp, MtpState
from .target import Target, TargetConfig, TargetState

__all__ = ["Draft", "DraftConfig", "DraftState", "DSpine", "DSpineConfig", "Dv", "DvConfig", "Eagle4", "Eagle4Config",
           "LlamaConfig", "Target", "TargetConfig", "TargetState", "greedy_plain", "greedy_spec", "greedy_spec_eagle4",
           "greedy_spec_mtp", "Mtp", "MtpState"]

"""Tests for the top-level package and data utilities."""

import numpy as np
import pytest

import finflowrl


class TestPackage:
    def test_version(self):
        assert hasattr(finflowrl, "__version__")
        assert isinstance(finflowrl.__version__, str)

    def test_imports(self):
        from finflowrl.env import MarketSimulator, HFTEnv
        from finflowrl.experts import Expert, AvellanedaStoikovExpert
        from finflowrl.models import MeanFlowPolicy, FiLMLayer, NoisePolicy
        from finflowrl.evaluation import evaluate_strategy
        from finflowrl.training import MeanFlowPretrainer, FlowRLFinetuner
        from finflowrl.utils import Config

    def test_docstring(self):
        assert finflowrl.__doc__ is not None

"""
Experimental code — NOT part of the tested surface of deep-fbsde-nn.

Only greeks_viz.py remains here (BS/BSB price/delta-vs-spot plotting,
unvalidated). exp_xva.py graduated in v0.2 after its price and component
breakdown were validated against the classical Monte-Carlo oracle — see
tests/test_xva.py.
"""

import warnings

warnings.warn(
    "experiments.experimental is untested code — currently only the BS/BSB "
    "greeks plotting helpers. Not part of the supported surface.",
    UserWarning,
    stacklevel=2,
)

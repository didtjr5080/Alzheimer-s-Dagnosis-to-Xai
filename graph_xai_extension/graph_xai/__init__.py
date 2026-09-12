"""Standalone Graph XAI extension for the CLIPtoXAI Alzheimer MRI project.

This package is entirely additive: it reads the pre-existing clip_xai_app /
handoff pipeline through adapters and never imports anything that mutates
those files. See ../docs/INTEGRATION_GUIDE.md for how a maintainer could wire
this into the existing app by hand.
"""

__version__ = "0.2.0"

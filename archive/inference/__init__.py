"""Inference adapter package for the NEURON biophysical-network project.

Exposes :mod:`inference.adapter`, which runs the vendored LIF inference
(``inference/lif_inference/``) on NEURON-generated sessions and reports AUC/FDR.
The vendored package is a verbatim copy of the LIF project's ``lif_inference``
(see ``lif_inference/SOURCE.md``); the LIF project remains the source of truth.
"""

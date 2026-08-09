"""Hardware-Aware Gateway: profile-driven kernel work on LLM inference.

The package is deliberately split so that nothing imports a backend it does not
need. Importing `hag` on a machine with no CUDA and no Metal is expected to work
and is covered by the test suite.
"""

__version__ = "0.1.0"

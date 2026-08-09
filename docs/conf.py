"""
Sphinx configuration for the GyRAPH documentation.

Copyright © 2026 Chun Hei Michael Chan, MIPLab EPFL
"""

import os
import sys
import types
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

# -- Stubs for heavy optional dependencies -----------------------------------
#
# torch and friends are only needed for the gradient-descent fitting helpers
# and the video/mesh utilities, so Read the Docs does not install them.
#
# ``autodoc_mock_imports`` alone is not enough for torch: SciPy introspects
# ``torch.Tensor`` with ``issubclass``, which raises on a Sphinx mock object and
# makes ``import gyraph`` fail outright. So torch is registered here as a stub
# module whose attributes are genuine classes; the rest go through the usual
# mock (see ``autodoc_mock_imports`` below).


class _StubMeta(type):
    """Metaclass whose attribute access yields further stub classes."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _StubMeta(name, (), {})

    def __call__(cls, *args, **kwargs):  # e.g. torch.tensor(...)
        return _StubMeta(cls.__name__, (), {})


class _StubModule(types.ModuleType):
    """Importable placeholder exposing a real class for any attribute."""

    __path__: list = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        stub = _StubMeta(name, (), {})
        setattr(self, name, stub)
        return stub


for _name in ("torch", "torch.nn", "torch.optim"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = _StubModule(_name)

# -- Project information -----------------------------------------------------

project = "GyRAPH"
author = "Chun Hei Michael Chan"
copyright = f"{date.today().year}, {author}, MIP:Lab, EPFL"

try:
    from gyraph import __version__ as release  # noqa: E402
except Exception:  # pragma: no cover - fallback when the package cannot import
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
master_doc = "index"

# Not installed on Read the Docs; see the stub section at the top of this file.
autodoc_mock_imports = [
    "torchvision",
    "torch_geometric",
    "cv2",
    "skimage",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"
autosummary_generate = True

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True

nitpicky = False
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_title = f"GyRAPH {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
    "titles_only": False,
    "style_external_links": True,
}

html_context = {
    "display_github": True,
    "github_user": "miki998",
    "github_repo": "GyRAPH",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

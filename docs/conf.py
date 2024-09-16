"""Configuration file for the Sphinx documentation builder."""

import sys
import typing
from pathlib import Path

if "doctest" not in sys.argv:  # Avoid type checking/isinstance failures.
    # Tag used to avoid expanding arraylike alias in docs
    typing.GENERATING_DOCUMENTATION = True


sys.path.insert(0, Path("..").resolve())

project = "FlowJax"
copyright = "2022, Daniel Ward"
author = "Daniel Ward"

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_css_files = ["style.css"]

html_theme_options = {
    "use_fullscreen_button": False,
    "use_download_button": False,
    "use_repository_button": True,
    "repository_url": "https://github.com/danielward27/flowjax",
    "home_page_in_toc": True,
}

html_title = "FlowJAX"

pygments_style = "xcode"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

napolean_use_rtype = False
napoleon_attr_annotations = True
napoleon_use_ivar = True

add_module_names = False
autodoc_inherit_docstrings = True
python_maximum_signature_line_length = 88

"""Configuration file for the Sphinx documentation builder."""

import sys
import typing
from pathlib import Path

# tag used in jaxtyping prevent expanding cumbersome type aliases such as ArrayLike
typing.GENERATING_DOCUMENTATION = True


sys.path.insert(0, Path("..").resolve())

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FlowJax"
copyright = "2022, Daniel Ward"
author = "Daniel Ward"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

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

# napoleon_include_init_with_doc = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_css_files = [
    "style.css",
]

html_theme_options = {
    "navigation_depth": 2,
}

pygments_style = "xcode"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

napolean_use_rtype = False
napoleon_attr_annotations = True

add_module_names = False

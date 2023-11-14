"""Configuration file for the Sphinx documentation builder."""
import sys
from pathlib import Path

import jax  # noqa Required to avoid circular import

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
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "nbsphinx",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

add_module_names = False
napoleon_include_init_with_doc = False

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
autodoc_typehints = "none"
autodoc_member_order = "bysource"

copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# Configuration file for the Sphinx documentation builder.
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FlowJax'
copyright = '2022, Daniel Ward'
author = 'Daniel Ward'
release = 'v6.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.todo", "sphinx.ext.viewcode", "sphinx.ext.autodoc",
    "sphinx.ext.napoleon", "sphinx.ext.doctest", "nbsphinx", "sphinx_copybutton"
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = [
    'style.css',
]

napoleon_include_init_with_doc = True

html_theme_options = {
    'navigation_depth': 2,
    'logo_only': True,
}

# html_logo = "../images/flowjax_logo.png"
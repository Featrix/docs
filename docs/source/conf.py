# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'Featrix'
copyright = '2023, Featrix, Inc.'
author = 'Featrix'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon'
]

# Napolean settings
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
#
# other stuff documented here - https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'

project = "NTX"
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_static/manuscript_claims.md",
    "_static/manuscript_tables.md",
]
html_theme = "sphinx_rtd_theme"
myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath"]
myst_heading_anchors = 3

"""Module to aid in creating good looking plots for the use in LaTeX documents.

Written by Aaron Young.
"""
# SciPy packages
import numpy as np
import pandas as pd
import matplotlib as mpl
import os.path
import math

FIGURE_PATH = '../figure'

mpl.use('pgf')

# %matplotlib inline

DEFAULT_FIGURE_SIZE = 0.9


# Function to calculate figure size in LaTeX document
def figsize(scale):
    """Determine a good size for the figure given a scale."""
    fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    if scale < 0.7:
        golden_mean *= 1.2
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


# pgf settings for use in LaTeX
pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(DEFAULT_FIGURE_SIZE),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
mpl.rcParams.update(pgf_with_latex)

# Import pyplot after mpl.use() is set
import matplotlib.pyplot as plt


# Custom newfig and save fig commands
def newfig(width=DEFAULT_FIGURE_SIZE):
    """Create a new figure."""
    # plt.clf()
    fig = plt.figure(figsize=figsize(width))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename):
    """Save a figure to a file."""
    # pgf_path = os.path.join(FIGURE_PATH, '{}.pgf'.format(filename))
    pdf_path = os.path.join(FIGURE_PATH, '{}.pdf'.format(filename))

    # Make sure the distination exists
    # os.makedirs(os.path.dirname(pgf_path), exist_ok=True)
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)

    # Make sure the label fits on the figure
    plt.tight_layout()

    # Save the figure
    # plt.savefig(pgf_path)
    plt.savefig(pdf_path)

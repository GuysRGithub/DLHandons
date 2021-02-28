from IPython import display
from matplotlib import pyplot as plt


def use_svg_display():
    """Use the svg format to display plot in jupyter."""
    display.set_matplotlib_formats('svg')

#!/usr/bin/env python
"""
--------------------------------
project: code
created: 12/04/2018 12:47
---------------------------------

"""


def rc(kwds={}):
    default_kwds = {
        'figure.figsize': (20, 10),
        'font.size': 12
    }
    default_kwds.update(kwds)
    return kwds


def multi_ax_legend(*ax, **kwargs):
    ax = list(ax)
    ax0 = ax.pop(0)
    lins, labs = ax0.get_legend_handles_labels()
    for a in ax:
        morelins, morelabs = a.get_legend_handles_labels()
        lins.extend(morelins)
        labs.extend(morelabs)
    return ax0.legend(lins, labs, **kwargs)


def savefig(fig, path, **kwargs):
    kws = dict(
            # dpi=1000,
            # format="eps",
            bbox_inches="tight"
    )
    kws.update(kwargs)
    return fig.savefig(
            path,
            **kws
    )

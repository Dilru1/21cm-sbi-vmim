# _palette.py
"""Shared arm colours + figure format for the stage-4 comparison plots.

Okabe-Ito: the standard 8-colour qualitative palette that stays distinguishable
under deuteranopia, protanopia and tritanopia, and survives greyscale printing.
Reference: Okabe & Ito (2008), "Color Universal Design".

Ordered so the first entries are the highest-contrast pairs -- with the usual
2-4 arms you get blue / vermillion / green / reddish-purple, unambiguous under
every common form of colour vision deficiency.

Yellow is last on purpose: legible as a line, but it washes out in the
alpha-blended contourf fills the corner plot uses. Black is excluded entirely --
see the note by the tuple.
"""

OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
)
# Black is deliberately EXCLUDED: both plots already reserve black for
# non-arm content (the SBC bias / under-confidence reference curves, and the
# truth lines in the corner plot), so an arm drawn in black would be read as
# one of those. Seven arms fit without repeating; beyond that the colour
# cycles and the line style advances.

# Line styles cycle more slowly than colours, so arms 9+ stay separable and the
# figure still reads correctly if printed in greyscale.
LINESTYLES = ("-", "--", "-.", ":")

FIGFMT = "pdf"   # publication output for corner + sbc


def arm_color(i):
    """Colour for arm index i, cycling past 8 arms."""
    return OKABE_ITO[i % len(OKABE_ITO)]


def arm_linestyle(i):
    """Line style for arm index i; changes only after a full colour cycle."""
    return LINESTYLES[(i // len(OKABE_ITO)) % len(LINESTYLES)]

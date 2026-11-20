"""Former figure of application domains (submarine, data center, LiDAR, fiber sensing) - removed in revision 3.

The applications figure was deleted from the manuscript (revision 3: the fiber-sensing and LiDAR material was
removed or shortened, Reviewer 2, point M3). This script is kept as a no-op so that existing run lists do not
fail. It imports nothing from the other figure scripts and produces no figure. If an output of the former
version (FigApplications.png / .pdf) is still present in ../figures, it is deleted so that it cannot be
mistaken for a current figure.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))
LEGACY_OUTPUTS = ("FigApplications.png", "FigApplications.pdf")


def main():
    removed = []
    for fn in LEGACY_OUTPUTS:
        p = os.path.join(FIGDIR, fn)
        if os.path.exists(p):
            os.remove(p)
            removed.append(fn)
    print("fig_applications.py: the applications figure was removed in revision 3; nothing is generated.")
    if removed:
        print("removed outdated output(s):", ", ".join(removed))


if __name__ == "__main__":
    main()

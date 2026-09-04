"""`python -m deep_fbsde_nn` — version info and citation helper."""

import argparse

import deep_fbsde_nn

BIBTEX = """@software{nodis2026deepfbsde,
  author = {Nodis, Ionut},
  title = {Deep FBSDE Neural Networks},
  year = {2026},
  doi = {10.5281/zenodo.22311423},
  url = {https://github.com/ionutnodis/deep-fbsde-nn}
}"""


def main():
    parser = argparse.ArgumentParser(
        prog="python -m deep_fbsde_nn",
        description="deep-fbsde-nn: Deep BSDE solvers for high-dimensional PDEs",
    )
    parser.add_argument(
        "--cite", action="store_true", help="print the BibTeX citation and exit"
    )
    args = parser.parse_args()

    if args.cite:
        print(BIBTEX)
        return

    print(f"deep-fbsde-nn {deep_fbsde_nn.__version__}")
    print("docs: https://github.com/ionutnodis/deep-fbsde-nn")
    print("cite: python -m deep_fbsde_nn --cite")


if __name__ == "__main__":
    main()

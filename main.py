"""
Simple entry point demonstrating package import.

For full usage, see the experiments/ directory.
"""

from deep_fbsde_nn import networks, equations, solvers, utils


def main():
    print("deep-fbsde-nn package loaded successfully!")
    print(f"Available networks: {networks.__all__}")
    print(f"Available equations: {equations.__all__}")
    print(f"Available solvers: {solvers.__all__}")


if __name__ == "__main__":
    main()

# Benchmark results

torch 2.14.0 · macOS-26.5.2-arm64-arm-64bit · CPU

| Equation | d | Solver | Reference | Rel. error | Time |
|---|---|---|---|---|---|
| Black-Scholes-Barenblatt | 2 | Standard | exact solution | 1.89% | 5s |
| Black-Scholes-Barenblatt | 10 | Stepwise | exact solution | 0.11% | 15s |
| Black-Scholes-Barenblatt | 50 | Stepwise | exact solution | 0.12% | 32s |
| Black-Scholes-Barenblatt | 100 | Stepwise | exact solution | 0.06% | 48s |
| Vanilla call | 1 | Standard | Black-Scholes closed form | 1.01% | 27s |
| BS basket call | 5 | Stepwise | seeded MC (same payoff) | 0.62% | 27s |
| BS basket call | 25 | Stepwise | seeded MC (same payoff) | 0.35% | 36s |
| Hamilton-Jacobi-Bellman | 3 | Stepwise | Cole-Hopf MC (seeded) | 2.31% | 22s |
| Hamilton-Jacobi-Bellman | 20 | Stepwise | Cole-Hopf MC (seeded) | 0.28% | 44s |
| Hamilton-Jacobi-Bellman | 100 | Stepwise | Cole-Hopf MC (seeded) / published 4.5901 | 0.03% | 102s |
| Allen-Cahn | 100 | Stepwise | published 0.052802 (branching diffusion) | 0.98% | 103s |

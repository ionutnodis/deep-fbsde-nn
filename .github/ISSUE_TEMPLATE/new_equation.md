---
name: New equation proposal
about: Propose (or contribute) a new PDE/FBSDE equation class
labels: enhancement, good first issue
---

**The PDE** (in the form ∂ₜu + μ·∇u + ½Tr(σσᵀD²u) + f = 0):

**Why it's interesting** (application, paper, benchmark):

**Reference solution available?**
- [ ] Exact / closed form
- [ ] Monte-Carlo formula (like HJB's Cole-Hopf reference)
- [ ] Literature value only
- [ ] None (ships contract-tested, like Allen-Cahn today)

See "Extending the Library" in the README — a subclass needs `drift`,
`diffusion`, `driver` (the f above; sign convention documented there),
and `terminal`.

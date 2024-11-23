# PalaceForCQED

EM simulations workflow scripts for superconducting quantum circuits using [awslabs/palace](https://github.com/awslabs/palace).

## Example

![](examples/imgs/circuit.png)

### Running Simulations

```bash
cd PalaceForCQED
julia --project=. examples/run.jl
```

### Convergence Test

See `examples/convergence.jl` and `examples/convergence_plot.py` for details.

![](examples/imgs/convergence_test_qubit.svg)

![](examples/imgs/convergence_test_resonator.svg)

### Port Response

See `examples/driven.jl` and `examples/driven_plot.py` for details.

![](examples/imgs/lumped_7.0-7.8_Step0.001_2024-10-25T000112.svg)
![](examples/imgs/lumped_4.36-4.42_Step0.0001_2024-10-25T111952.svg)

### Lj Sweep

See `examples/sweep_Lj.jl` and `examples/sweep_Lj_plot.py` for details.
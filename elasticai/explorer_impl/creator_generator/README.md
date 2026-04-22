# Creator-Generator

The creator-Generator is a implementation the elastic-ai.Explorer Generator classes. It enables deployment and measurements of hardware metrics on enV5 and simulation using GHDL and Cocotb.
The general setup is explained in the top-level [README.md](../../../README.md) under "Creator-Generator Dependencies" and "Set up your enV5 for Deployment".

It is recommended to use the explorer.Explorer class to use the Generator. A extensive example of this is found in creator_example.py. 

# Simulation
This rely on the feature to register callables to metrics in metric_to_source. These are Called MetricFunctions with the type `Callable[[Host, "HWManager", Path | None], dict[str, dict]]`

Example:
```python
metric_to_source = {
    Metric.ACCURACY: _some_metric_function
    }
```

In this MetricFunctions we have the complete freedom to execute test in simulation and then return the results as a {"metric_name": {"value": value, "unit": unit}} similar to the other generator implementations. An example for this are the functions `_run_accuracy_simulation` and `_run_latency_simulation` in  `creator_example.py`. 

# On device (beta)
To deploy a model to the enV5, it has to be synthesized for the specific hardware platform 'env5_s50' or 'env5_s15'. This is parametrized in the VivadoParams.
To allow Synthesis without having to locally install the Vivado Design Suite, the synthesis can be run on the Hetzner Server of the IES Duisburg Essen.
The access has to be requested individually from the contributors.

Example:
```python
compiler_params = VivadoParams(
        "/home/vivado/<username>-build/", "<ip-address>", "vivado", "<hardware_platform>"
    )
```

An example for a MetricFunction for measurements on device is the `_run_accuracy_deployed function` in `creator_example.py`. 

# Limitations on model architectures
Currently only single layered models give stable results, the layers of the creator.fixed_point package cannot learn functions probably when chained together.

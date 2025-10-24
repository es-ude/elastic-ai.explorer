from dataclasses import asdict, is_dataclass
from typing import Dict

from elasticai.explorer.hw_nas.optimization_criteria import OptimizationCriteriaRegistry


def dataclass_instance_to_toml(
    instance: object,
    name: str | None = None,
    indent: int = 0,
    additional_info: Dict = {},
) -> str:

    if not is_dataclass(instance):
        raise TypeError("to_toml_block expects a dataclass instance")

    cls_name = name or type(instance).__name__
    pad = " " * indent
    d = asdict(instance)  # type: ignore

    lines = [f"{pad}{cls_name}:"]
    for k, v in d.items():
        lines.append(f"{pad}    {k}: {v}")

    for k, v in additional_info.items():
        lines.append(f"{pad}    {k}: {v}")
    return "\n".join(lines)


def opt_crit_registry_to_toml(reg: OptimizationCriteriaRegistry) -> str:
    chunks: list[str] = []

    for est in reg.get_estimators():
        est_key = getattr(est, "__name__", repr(est))
        chunks.append(f"{est_key}:")
        for entry in reg.get_criteria(est):
            block = dataclass_instance_to_toml(
                entry, name=entry.__class__.__name__, indent=4
            )
            chunks.append(block)

    return "\n".join(chunks)

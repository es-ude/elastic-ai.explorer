from abc import ABC, abstractmethod
import inspect
from optuna.samplers import (
    TPESampler,
    RandomSampler,
    CmaEsSampler,
    BaseSampler,
    BruteForceSampler,
    GPSampler,
    QMCSampler,
    NSGAIISampler,
    NSGAIIISampler,
)

class SamplerBuilder(ABC):
    """Abstract base class for sampler builders."""

    def _get_valid_params(self, sampler: type[BaseSampler], **kwargs):
        sig = inspect.signature(sampler.__init__)
        valid_params = set(sig.parameters.keys()) - {"self"}
        params = {k: v for k, v in kwargs.items() if k in valid_params}
        return params

    @abstractmethod
    def build(self, **kwargs) -> BaseSampler:
        """Build and return a sampler."""
        pass


class TPESamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(TPESampler, **kwargs)
        return TPESampler(**params)


class RandomSamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(RandomSampler, **kwargs)
        return RandomSampler(**params)

class CmaEsSamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(CmaEsSampler, **kwargs)
        return CmaEsSampler(**params)


class BruteForceSamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(BruteForceSampler, **kwargs)
        return BruteForceSampler(**params)


class GPSamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(GPSampler, **kwargs)
        return GPSampler(**params)


class QMCSamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(QMCSampler, **kwargs)
        return QMCSampler(**params)


class NSGAIISamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(NSGAIISampler, **kwargs)
        return NSGAIISampler(**params)


class NSGAIIISamplerBuilder(SamplerBuilder):
    def build(self, **kwargs) -> BaseSampler:
        params = self._get_valid_params(NSGAIIISampler, **kwargs)
        return NSGAIIISampler(**params)


def get_sampler(sampler_type: str, **kwargs):
    """Factory function to get a sampler by type."""
    builders = {
        "tpe": TPESamplerBuilder(),
        "random": RandomSamplerBuilder(),
        "cmaes": CmaEsSamplerBuilder(),
        "bruteforce": BruteForceSamplerBuilder(),
        "gp": GPSamplerBuilder(),
        "qmc": QMCSamplerBuilder(),
        "nsgaii": NSGAIISamplerBuilder(),
        "nsgaiii": NSGAIIISamplerBuilder(),
    }

    if sampler_type not in builders:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    return builders[sampler_type].build(**kwargs)

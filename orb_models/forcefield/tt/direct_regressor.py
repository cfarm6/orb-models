"""TT-accelerated wrapper around direct Orb forcefield regressors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

from orb_models.common.atoms.batch.graph_batch import AtomGraphs
from orb_models.forcefield import pretrained
from orb_models.forcefield.forcefield_adapter import ForcefieldAtomsAdapter
from orb_models.forcefield.models.direct_regressor import DirectForcefieldRegressor
from orb_models.forcefield.tt.backend import TTBackend, TTDevice
from orb_models.forcefield.tt.modules import patch_linears

DirectModelLoader = Callable[..., tuple[DirectForcefieldRegressor, ForcefieldAtomsAdapter]]
# Direct-only pretrained loaders exposed for TT inference.
DIRECT_PRETRAINED_MODELS: dict[str, DirectModelLoader] = {
    name: cast(DirectModelLoader, loader)
    for name, loader in pretrained.ORB_PRETRAINED_MODELS.items()
    if name.startswith("orb-v3-direct")
    or name.startswith("orbmol-v1-direct")
    or name.startswith("separate-d")
    or name in {"orb-v2", "orb-mptraj-only-v2", "orb-d3-v2", "orb-d3-sm-v2", "orb-d3-xs-v2"}
}


def load_tt_direct_pretrained(
    model_name: str,
    *,
    backend: TTBackend | Literal["auto"] = "auto",
    device_id: int = 0,
    precision: str = "float32-high",
    compile: bool | None = False,
    **loader_kwargs,
) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
    """Load a direct pretrained Orb model and wrap it for TT inference.

    Args:
        model_name: Key from :data:`DIRECT_PRETRAINED_MODELS` / ``pretrained.ORB_PRETRAINED_MODELS``.
        backend: ``simulator``, ``hardware``, or ``auto``.
        device_id: Tenstorrent device index.
        precision: Floating point precision passed to the CPU model loader.
        compile: Whether to torch.compile the base model on CPU before patching.
        loader_kwargs: Extra kwargs forwarded to the pretrained loader.
    """

    try:
        loader = DIRECT_PRETRAINED_MODELS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(DIRECT_PRETRAINED_MODELS))
        raise ValueError(
            f"Unknown direct pretrained model {model_name!r}. Supported: {supported}"
        ) from exc
    return TTDirectForcefieldRegressor.from_pretrained(
        loader,
        backend=backend,
        device_id=device_id,
        precision=precision,
        compile=compile,
        **loader_kwargs,
    )


class TTDirectForcefieldRegressor:
    """Accelerates a CPU ``DirectForcefieldRegressor`` by offloading linears to TT."""

    def __init__(self, model: DirectForcefieldRegressor, tt_device: TTDevice) -> None:
        self.model = model
        self.tt_device = tt_device
        self._patched = False

    @classmethod
    def from_pretrained(
        cls,
        loader: Callable[..., tuple[DirectForcefieldRegressor, ForcefieldAtomsAdapter]],
        *,
        backend: TTBackend | Literal["auto"] = "auto",
        device_id: int = 0,
        precision: str = "float32-high",
        compile: bool | None = False,
        **loader_kwargs,
    ) -> tuple[TTDirectForcefieldRegressor, ForcefieldAtomsAdapter]:
        """Load a direct pretrained model on CPU and wrap it for TT inference."""

        model, adapter = loader(
            device="cpu",
            precision=precision,
            compile=compile,
            train=False,
            **loader_kwargs,
        )
        if not isinstance(model, DirectForcefieldRegressor):
            raise TypeError("TT acceleration is only supported for direct forcefield models.")
        tt_device = TTDevice(backend=backend, device_id=device_id)
        return cls(model=model, tt_device=tt_device), adapter

    def _ensure_patched(self) -> None:
        if self._patched:
            return
        self.tt_device.open()
        patch_linears(self.model, self.tt_device.runner)
        self._patched = True

    def forward(self, batch: AtomGraphs):
        self._ensure_patched()
        return self.model(batch)

    def predict(self, batch: AtomGraphs, **kwargs):
        self._ensure_patched()
        return self.model.predict(batch, **kwargs)

    @property
    def properties(self):
        return self.model.properties

    def close(self) -> None:
        self.tt_device.close()

    def clear_profile(self) -> None:
        self.tt_device.runner.clear_profile()

    def profile_summary(self) -> dict[str, dict[str, float | int]]:
        return self.tt_device.runner.profile_summary()

    def __getattr__(self, name: str):
        return getattr(self.model, name)

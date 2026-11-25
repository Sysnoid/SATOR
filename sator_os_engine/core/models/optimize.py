from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..optimizer.utils import KNOWN_ACQUISITIONS, KNOWN_GOALS, normalize_acquisition_name, normalize_goal

MAX_BATCH = 16


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str = Field("qnehvi")
    # Multi-objective (min/max): qnehvi=qLogEHVI (default), qehvi, qnoisyehvi, parego|qei, qpi, qucb|ucb.
    # Single-objective (min/max): same names map to qLogEI, qEI, qPI, qUCB, etc. Use acquisition="sobol" to force the legacy Sobol+scoring grid.
    acquisition: str = Field("qnehvi")
    # lab: exact GP and, when acquisition is "auto", a noisy acquisition.
    # throughput: "auto" stays on qLogEI / qLogEHVI unless Yvar, replicates, or failures are present.
    mode: str = Field("lab")
    batch_size: int = Field(4, ge=1, le=MAX_BATCH)
    max_evaluations: int = Field(100, ge=1)
    seed: int | None = None
    use_pca: bool = False
    pca_dimension: int | str | None = None
    # Expands the PCA search box from the training envelope [0, 1] to [-m, 1+m].
    pca_margin: float = Field(0.0, ge=0.0, le=2.0)
    # training: keep candidates within the training off-subspace residual.
    # none: do not apply that cap. A number is an explicit residual cap.
    pca_trust: str | float = "training"
    parameter_scaling: str | None = None  # none|standardize|minmax
    value_normalization: str | None = None  # none|standardize|minmax
    target_tolerance: float | None = None
    target_variance_penalty: float | None = None
    # Hard-constraint goal enforcement (enforce_above / enforce_below /
    # enforce_within_range). Feasibility of the GP posterior against the
    # threshold is evaluated on ``mu`` by default. Setting a positive margin
    # switches enforcement to a confidence bound: for enforce_above the
    # lower bound ``mu - margin * sigma`` must satisfy the threshold;
    # for enforce_below it is ``mu + margin * sigma``; for
    # enforce_within_range both bounds must lie inside the window.
    # Typical choices: 0.0 (default, mean only), 1.0 (~1 sigma),
    # 2.0 (~95% confidence). See docs/06-objectives-and-constraints.md.
    enforcement_uncertainty_margin: float = 0.0
    sum_constraints: list[dict[str, Any]] | None = None  # [{"indices":[0,1,2],"target_sum":1.0}]
    ratio_constraints: list[dict[str, Any]] | None = None  # [{"i":0,"j":1,"min_ratio":0.5,"max_ratio":2.0}]
    # {"max_nonzero": 4, "min_dose": 0.02} applied inside each sum group.
    ingredient_rules: dict[str, Any] | None = None
    # none leaves ingredients in the original coordinates. ilr maps each sum group onto the simplex.
    mixture_transform: str | None = None
    # GP surface/volume maps for visualization (2D/3D only)
    return_maps: bool = False
    map_space: str = Field("input")  # input|pca
    map_resolution: list[int] | None = None  # [nx,ny,(nz)]
    # Advanced numerical settings (optional). Read keys: num_restarts, raw_samples,
    # acq_maxiter, batch_limit, sequential, ref_point.
    advanced: dict[str, Any] | None = None
    # Optional GP configuration and acquisition parameters
    # lengthscale|outputscale|noise: applied before MLL fit as hints. If fix_lengthscale|fix_outputscale|fix_noise
    # is true, that value is re-applied after fit and the parameter is frozen (not overwritten by optimization).
    gp_config: dict[str, Any] | None = None
    acquisition_params: dict[str, Any] | None = None  # e.g., {"ucb_beta": 0.2, "raw_samples": 16, "ref_point": [...]}
    outputs: dict[str, Any] = Field(default_factory=dict)
    # gp: exact ARD GP. saas_map: MAP sparse prior. saas: NUTS sparse prior. kpls: PLS kernel.
    model: str = "gp"
    # direct searches the full box. trust_region searches the replayed TuRBO box.
    route: str = "direct"

    @model_validator(mode="after")
    def _known_names(self) -> OptimizationConfig:
        acq = normalize_acquisition_name(self.acquisition)
        if acq not in KNOWN_ACQUISITIONS:
            raise ValueError(f"unknown acquisition {self.acquisition!r}")
        algo = normalize_acquisition_name(self.algorithm)
        if self.algorithm and algo not in KNOWN_ACQUISITIONS:
            raise ValueError(f"unknown algorithm {self.algorithm!r}")
        for label, value in (
            ("parameter_scaling", self.parameter_scaling),
            ("value_normalization", self.value_normalization),
        ):
            if value is None:
                continue
            if str(value).strip().lower() not in {"none", "standardize", "minmax"}:
                raise ValueError(f"{label} must be one of none, standardize, minmax")
        if str(self.mode).strip().lower() not in {"lab", "throughput"}:
            raise ValueError("mode must be 'lab' or 'throughput'")
        transform = None if self.mixture_transform is None else str(self.mixture_transform).strip().lower()
        if transform not in {None, "none", "ilr"}:
            raise ValueError("mixture_transform must be 'none' or 'ilr'")
        if transform == "ilr" and self.use_pca:
            raise ValueError("mixture_transform 'ilr' cannot be combined with use_pca")
        rules = self.ingredient_rules
        if rules is not None:
            if rules.get("max_nonzero") is not None and int(rules["max_nonzero"]) < 1:
                raise ValueError("ingredient_rules.max_nonzero must be >= 1")
            if rules.get("min_dose") is not None and float(rules["min_dose"]) < 0:
                raise ValueError("ingredient_rules.min_dose must be >= 0")
        if self.use_pca and self.pca_dimension is None:
            raise ValueError("pca_dimension is required when use_pca is true")
        if isinstance(self.pca_dimension, str) and self.pca_dimension.strip().lower() != "auto":
            raise ValueError("pca_dimension must be an integer or 'auto'")
        trust = self.pca_trust
        if isinstance(trust, str):
            if trust.strip().lower() not in {"training", "none"}:
                raise ValueError("pca_trust must be 'training', 'none', or a non-negative number")
        elif float(trust) < 0.0:
            raise ValueError("pca_trust must be 'training', 'none', or a non-negative number")
        model_name = str(self.model or "gp").strip().lower()
        route_name = str(self.route or "direct").strip().lower()
        self.model = model_name
        self.route = route_name
        if model_name not in {"gp", "saas_map", "saas", "kpls"}:
            raise ValueError("model must be 'gp', 'saas_map', 'saas', or 'kpls'")
        if route_name not in {"direct", "trust_region"}:
            raise ValueError("route must be 'direct' or 'trust_region'")
        if model_name in {"saas", "saas_map", "kpls"} and self.use_pca:
            raise ValueError(f"model {model_name!r} cannot be combined with use_pca")
        if model_name in {"saas", "saas_map"} and transform == "ilr":
            raise ValueError(f"model {model_name!r} cannot be combined with mixture_transform 'ilr'")
        if route_name == "trust_region" and self.use_pca:
            raise ValueError("route 'trust_region' cannot be combined with use_pca")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: dict[str, Any]
    search_space: dict[str, Any]
    objectives: dict[str, Any]
    constraints: dict[str, Any] | None = None
    optimization_config: OptimizationConfig

    @model_validator(mode="after")
    def _validate_problem(self) -> OptimizeRequest:
        self._merge_request_constraints()
        params = self.search_space.get("parameters") if isinstance(self.search_space, dict) else None
        if not isinstance(params, list) or not params:
            raise ValueError("search_space.parameters must be a non-empty list")
        self._validate_parameters(params)
        transform = str(self.optimization_config.mixture_transform or "none").lower()
        if transform == "ilr" and any(str(p.get("type", "float")) == "categorical" for p in params):
            raise ValueError("mixture_transform 'ilr' does not support categorical parameters")
        model_name = str(self.optimization_config.model or "gp").strip().lower()
        if model_name in {"saas", "saas_map"} and any(str(p.get("type", "float")) == "categorical" for p in params):
            raise ValueError(f"model {model_name!r} does not support categorical parameters")
        if not isinstance(self.objectives, dict) or not self.objectives:
            raise ValueError("objectives must be a non-empty object")
        for name, cfg in self.objectives.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"objectives.{name} must be an object")
            goal = normalize_goal(cfg.get("goal", ""))
            if goal not in KNOWN_GOALS:
                raise ValueError(f"objectives.{name}.goal: unknown goal {cfg.get('goal')!r}")
        x_rows = self.dataset.get("X") if isinstance(self.dataset, dict) else None
        y_rows = self.dataset.get("Y") if isinstance(self.dataset, dict) else None
        if not isinstance(x_rows, (list, tuple)) or not isinstance(y_rows, (list, tuple)):
            raise ValueError("dataset.X and dataset.Y are required")
        if len(x_rows) != len(y_rows) or len(x_rows) < 1:
            raise ValueError(
                f"len(dataset.X) must equal len(dataset.Y) and be >= 1 (got {len(x_rows)} and {len(y_rows)})"
            )
        n_params = len(params)
        for i, row in enumerate(x_rows):
            if not isinstance(row, (list, tuple)) or len(row) != n_params:
                raise ValueError(f"dataset.X[{i}] must have length {n_params}")
            self._validate_x_row(i, row, params)
        y = np.asarray(y_rows, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.ndim != 2:
            raise ValueError("dataset.Y must be a 2-D array")
        n_obj = len(self.objectives)
        if y.shape[1] != n_obj:
            raise ValueError(f"dataset.Y has {y.shape[1]} column(s) but {n_obj} objectives were declared")
        yvar = self.dataset.get("Yvar")
        if yvar is not None:
            variances = np.asarray(yvar, dtype=float)
            if variances.ndim == 1:
                variances = variances.reshape(-1, 1)
            if variances.shape != y.shape:
                raise ValueError(f"dataset.Yvar shape {variances.shape} must match dataset.Y shape {y.shape}")
            if np.any(np.isfinite(variances) & (variances < 0)):
                raise ValueError("dataset.Yvar must be non-negative")
        failed = self.dataset.get("failed")
        if failed is not None and len(failed) != len(x_rows):
            raise ValueError("dataset.failed must have one entry per row of dataset.X")
        for j, name in enumerate(self.objectives):
            col = y[:, j]
            if np.isfinite(col).any():
                continue
            kind = "inf" if np.isinf(col).all() else "NaN" if np.isnan(col).all() else "non-finite"
            raise ValueError(f"objectives.{name}: column is entirely {kind}")
        if self.optimization_config.use_pca:
            k = self.optimization_config.pca_dimension
            assert k is not None
            finite_x = 0
            for row in x_rows:
                try:
                    vals = [float(v) for v in row]
                except (TypeError, ValueError):
                    continue
                if np.isfinite(vals).all():
                    finite_x += 1
            from ..optimizer.preprocess import resolve_pca_k

            resolve_pca_k(finite_x, n_params, k)
            numeric_rows = []
            for row in x_rows:
                try:
                    numeric_rows.append([round(float(value), 8) for value in row])
                except (TypeError, ValueError):
                    numeric_rows = []
                    break
            if len(numeric_rows) >= 2 and len({tuple(row) for row in numeric_rows}) < 2:
                raise ValueError("PCA components are degenerate: dataset.X rows are identical")
        return self

    def _merge_request_constraints(self) -> None:
        raw = self.constraints if isinstance(self.constraints, dict) else None
        if not raw:
            return
        cfg = self.optimization_config
        if not cfg.sum_constraints and raw.get("sum_constraints"):
            cfg.sum_constraints = list(raw["sum_constraints"])
        if not cfg.ratio_constraints and raw.get("ratio_constraints"):
            cfg.ratio_constraints = list(raw["ratio_constraints"])

    @staticmethod
    def _validate_parameters(params: list[Any]) -> None:
        names: set[str] = set()
        for p in params:
            if not isinstance(p, dict) or "name" not in p:
                raise ValueError("each search_space parameter needs a name")
            name = str(p["name"])
            if name in names:
                raise ValueError(f"duplicate parameter name {name!r}")
            names.add(name)
            ptype = str(p.get("type", "float"))
            if ptype not in {"float", "int", "categorical"}:
                raise ValueError(f"search_space.parameters.{name}: unsupported type {ptype!r}")
            if ptype == "categorical":
                choices = p.get("choices")
                if not isinstance(choices, (list, tuple)) or len(choices) < 1:
                    raise ValueError(f"search_space.parameters.{name}: categorical parameters need choices")
                continue
            if "min" not in p or "max" not in p:
                raise ValueError(f"search_space.parameters.{name}: min and max are required")
            lo, hi = float(p["min"]), float(p["max"])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                raise ValueError(f"search_space.parameters.{name}: min must be < max")

    @staticmethod
    def _validate_x_row(i: int, row: list[Any] | tuple[Any, ...], params: list[dict[str, Any]]) -> None:
        for j, param in enumerate(params):
            ptype = str(param.get("type", "float"))
            value = row[j]
            if ptype == "categorical":
                choices = [str(c) for c in param.get("choices") or []]
                if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    if int(value) < 0 or int(value) >= len(choices):
                        raise ValueError(f"dataset.X[{i}][{j}] is not a valid level of {param['name']!r}")
                elif str(value) not in choices:
                    raise ValueError(
                        f"dataset.X[{i}][{j}] value {value!r} is not one of {choices} for {param['name']!r}"
                    )
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"dataset.X[{i}][{j}] is not a number") from exc
            if not np.isfinite(number):
                kind = "inf" if np.isinf(number) else "NaN"
                raise ValueError(f"dataset.X[{i}][{j}] is {kind}")


class OptimizeResponse(BaseModel):
    job_id: str | None = None
    predictions: list[dict[str, Any]] | None = None
    pareto: dict[str, Any] | None = None
    encoding_info: dict[str, Any] | None = None  # e.g., pc_mins/maxs
    diagnostics: dict[str, Any] | None = None

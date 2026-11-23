"""Validate the speedup on saved draws from two frozen STF sweep bundles.

Both corrected paths use the same t* operator. An additional comparison to the
untouched historical model with attenuation disabled checks independence from
that intentional physics change. Inputs and output paths are explicit; source
runs are read only. Run in the TAPIR Python environment.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np


def relative_l2(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def assert_waveform_equal(actual, expected):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite forward output")
    np.testing.assert_allclose(
        actual, expected, rtol=1e-11,
        atol=1e-11 * max(float(np.max(np.abs(expected))), 1e-30),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-bundle", type=Path, required=True)
    parser.add_argument("--corrected-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws-per-chain", type=int, default=8)
    args = parser.parse_args()
    if args.draws_per_chain < 1:
        parser.error("--draws-per-chain must be positive")
    old = args.historical_bundle.resolve()
    new = args.corrected_bundle.resolve()
    sys.path.insert(0, str(new / "code"))
    import vespainv.waveformBuilder as corrected
    from vespainv.rjmcmc import compute_log_likelihood_L1
    from vespainv.utils import prep_data

    if not Path(corrected.__file__).resolve().is_relative_to(new / "code"):
        raise ValueError("Incorrect corrected package imported")
    spec = importlib.util.spec_from_file_location(
        "historical_waveform_builder", old / "code/vespainv/waveformBuilder.py"
    )
    historical = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(historical)
    results = []
    for case in json.loads((new / "cases.json").read_text()):
        observed, time, _, whitening, metadata, _ = prep_data(
            str(new / "data"), case["dataset"], True, "Z", 3,
        )
        for source, root in (("historical", old), ("corrected", new)):
            run = root / "runs" / case["dataset"] / case["runname"]
            bk = pickle.loads((run / "Bookkeeping.pkl").read_bytes())
            if not bk.fitAtts or not bk.fitPhase or bk.normOpt != 1:
                raise ValueError("This validation expects fitAtts, fitPhase and the L1 objective")
            stf = np.loadtxt(run / "stf_used.csv", delimiter=",", skiprows=1)
            no_atts = copy.copy(bk)
            no_atts.fitAtts = False
            errors, independent_errors, likelihood_errors = [], [], []
            for chain in sorted(run.glob("chain_*")):
                models = pickle.loads((chain / "ensemble.pkl").read_bytes())
                indices = np.linspace(0, len(models) - 1, min(args.draws_per_chain, len(models)), dtype=int)
                for index in indices:
                    model = models[index]
                    common = (model, metadata, time, stf[:, 0], stf[:, 1])
                    expected = corrected._create_U_from_model_3c_freqdomain_loop(*common, bk)
                    actual = corrected.create_U_from_model_3c_freqdomain(*common, bk)
                    assert_waveform_equal(actual, expected)
                    errors.append(relative_l2(actual, expected))
                    factors = np.asarray(whitening) * np.exp(-0.5 * model.loge)
                    ll_loop = compute_log_likelihood_L1(observed, expected, factors)
                    ll_fast = compute_log_likelihood_L1(observed, actual, factors)
                    np.testing.assert_allclose(ll_fast, ll_loop, rtol=1e-12, atol=1e-8)
                    likelihood_errors.append(abs(ll_fast - ll_loop))
                    independent = historical.create_U_from_model_3c_freqdomain(*common, no_atts)
                    fast_no_atts = corrected.create_U_from_model_3c_freqdomain(*common, no_atts)
                    assert_waveform_equal(fast_no_atts, independent)
                    independent_errors.append(relative_l2(fast_no_atts, independent))
            result = dict(
                case=case["case"], draw_source=source, models_checked=len(errors),
                maximum_corrected_loop_relative_L2=max(errors),
                maximum_likelihood_absolute_difference=max(likelihood_errors),
                maximum_unattenuated_historical_relative_L2=max(independent_errors),
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    report = dict(
        corrected_module=str(corrected.__file__),
        historical_module=str(historical.__file__),
        models_checked=sum(r["models_checked"] for r in results), cases=results,
        checks=["Corrected loop versus vectorized waveforms with attenuation and phase fitting",
                "Corresponding covariance-weighted L1 likelihood equality",
                "Untouched historical loop versus vectorized model with attenuation disabled"],
        limitation="Forward and likelihood equality validates vectorization; it does not prove chain convergence or equivalence to the old attenuated posterior.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

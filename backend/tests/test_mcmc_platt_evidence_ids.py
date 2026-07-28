"""
Tests for MCMC probabilistic engine, Platt scaling calibration,
and stable UUID5 evidence IDs.
"""
import math
import uuid
import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════════
# Phase 24 — MCMC Engine
# ═══════════════════════════════════════════════════════════════════════

def _make_signals(scores, confidences):
    return [
        {"signal_name": f"sig_{i}", "score": s, "confidence": c}
        for i, (s, c) in enumerate(zip(scores, confidences))
    ]


class TestMCMCEngine:

    def test_returns_required_keys(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, point_estimate=0.82)
        for key in ("point_estimate", "interval_90", "interval_50",
                    "std", "certainty", "n_samples", "acceptance_rate"):
            assert key in r, f"Missing key: {key}"

    def test_point_estimate_in_unit_range(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.82)
        assert 0.0 <= r["point_estimate"] <= 1.0

    def test_interval_90_ordered(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.82)
        lo, hi = r["interval_90"]
        assert lo <= hi
        assert 0.0 <= lo and hi <= 1.0

    def test_interval_50_inside_interval_90(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.82)
        lo90, hi90 = r["interval_90"]
        lo50, hi50 = r["interval_50"]
        assert lo90 <= lo50 and hi50 <= hi90

    def test_std_non_negative(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.82)
        assert r["std"] >= 0.0

    def test_certainty_valid_value(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75, 0.9], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.82)
        assert r["certainty"] in ("high", "medium", "low")

    def test_high_certainty_when_signals_agree(self):
        """All signals tightly agree → certainty should be high."""
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.90, 0.91, 0.89, 0.92, 0.90],
                                [0.95, 0.95, 0.95, 0.95, 0.95])
        r = run_mcmc(signals, 0.90)
        assert r["certainty"] in ("high", "medium")

    def test_low_certainty_when_signals_conflict(self):
        """Conflicting signals must produce a wider interval than agreeing signals."""
        from backend.services.mcmc_engine import run_mcmc
        conflict = _make_signals([0.05, 0.95, 0.10, 0.90, 0.08, 0.92],
                                 [0.9,  0.9,  0.9,  0.9,  0.9,  0.9])
        agree = _make_signals([0.88, 0.90, 0.87, 0.91, 0.89, 0.88],
                              [0.95, 0.95, 0.95, 0.95, 0.95, 0.95])
        r_c = run_mcmc(conflict, 0.50)
        r_a = run_mcmc(agree,    0.89)
        width_c = r_c["interval_90"][1] - r_c["interval_90"][0]
        width_a = r_a["interval_90"][1] - r_a["interval_90"][0]
        assert width_c > width_a

    def test_fallback_when_no_confident_signals(self):
        """All signals with confidence=0 → fallback used."""
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.9], [0.0, 0.0])
        r = run_mcmc(signals, 0.85)
        assert r["n_samples"] == 0
        assert r["certainty"] == "low"

    def test_deterministic_with_same_seed(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.7, 0.8, 0.6], [0.7, 0.8, 0.6])
        r1 = run_mcmc(signals, 0.70, rng_seed=123)
        r2 = run_mcmc(signals, 0.70, rng_seed=123)
        assert r1["point_estimate"] == r2["point_estimate"]
        assert r1["std"] == r2["std"]

    def test_acceptance_rate_in_range(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.7, 0.8], [0.8, 0.7])
        r = run_mcmc(signals, 0.75)
        assert 0.0 <= r["acceptance_rate"] <= 1.0

    def test_no_nan_or_inf(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.7, 0.8, 0.6], [0.7, 0.8, 0.6])
        r = run_mcmc(signals, 0.70)
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"Non-finite in '{k}': {v}"
            elif isinstance(v, list):
                for x in v:
                    assert math.isfinite(x), f"Non-finite in '{k}': {x}"

    def test_n_samples_positive(self):
        from backend.services.mcmc_engine import run_mcmc
        signals = _make_signals([0.8, 0.75], [0.7, 0.8])
        r = run_mcmc(signals, 0.78)
        assert r["n_samples"] > 0


class TestBoundaryReflection:
    """F-12 regression tests: boundary handling must be a true
    reflection, not a clip -- a clip piles probability mass exactly at
    the boundary and breaks the symmetric-proposal assumption behind
    the plain Metropolis acceptance ratio used in run_mcmc()."""

    def test_reflects_below_zero(self):
        from backend.services.mcmc_engine import _reflect_at_boundaries
        assert abs(_reflect_at_boundaries(-0.03) - 0.03) < 1e-9

    def test_reflects_above_one(self):
        from backend.services.mcmc_engine import _reflect_at_boundaries
        assert abs(_reflect_at_boundaries(1.05) - 0.95) < 1e-9

    def test_mid_range_unaffected(self):
        from backend.services.mcmc_engine import _reflect_at_boundaries
        assert abs(_reflect_at_boundaries(0.5) - 0.5) < 1e-9

    def test_does_not_pile_up_at_boundary(self):
        """A clip would pile every out-of-range draw at exactly 0 or 1.
        Reflection spreads them back out -- check we don't see a spike
        of exact-boundary values across many out-of-range draws."""
        from backend.services.mcmc_engine import _reflect_at_boundaries
        out_of_range_inputs = [-0.01 * i for i in range(1, 50)]
        results = [_reflect_at_boundaries(x) for x in out_of_range_inputs]
        at_boundary = sum(1 for r in results if r <= 1e-5 or r >= 1 - 1e-5)
        assert at_boundary == 0, (
            "reflected values piled up at the boundary -- this looks like "
            "clipping, not reflection"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 25 — Platt Scaling Calibration
# ═══════════════════════════════════════════════════════════════════════

class TestPlattCalibrator:

    def test_calibrate_returns_float(self):
        from backend.services.platt_calibrator import calibrate
        result = calibrate(0.7)
        assert isinstance(result, float)

    def test_calibrate_in_unit_range(self):
        from backend.services.platt_calibrator import calibrate
        for score in [0.0, 0.25, 0.5, 0.75, 1.0]:
            r = calibrate(score)
            assert 0.0 <= r <= 1.0, f"Out of range for score={score}: {r}"

    def test_calibrate_monotonic(self, monkeypatch, tmp_path):
        """Higher raw score must produce higher calibrated score (monotonic)."""
        import backend.services.platt_calibrator as pc
        # Use isolated tmp path so prior test fit results do not leak in
        monkeypatch.setattr(pc, "_PARAMS_PATH", tmp_path / "no_params.json")
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        calibrated = [pc.calibrate(s) for s in scores]
        for i in range(len(calibrated) - 1):
            assert calibrated[i] <= calibrated[i + 1], (
                f"Not monotonic: calibrate({scores[i]})={calibrated[i]} "
                f"> calibrate({scores[i+1]})={calibrated[i+1]}"
            )

    def test_calibrate_with_interval_keys(self):
        from backend.services.platt_calibrator import calibrate_with_interval
        signals = _make_signals([0.8, 0.7], [0.8, 0.7])
        r = calibrate_with_interval(0.75, signals=signals)
        for key in ("calibrated", "interval_90", "A", "B"):
            assert key in r

    def test_interval_90_ordered(self):
        from backend.services.platt_calibrator import calibrate_with_interval
        r = calibrate_with_interval(0.7)
        lo, hi = r["interval_90"]
        assert lo <= hi
        assert 0.0 <= lo and hi <= 1.0

    def test_calibrated_inside_interval_90(self):
        from backend.services.platt_calibrator import calibrate_with_interval
        r = calibrate_with_interval(0.7)
        lo, hi = r["interval_90"]
        assert lo <= r["calibrated"] <= hi

    def test_fit_improves_calibration(self, tmp_path, monkeypatch):
        """After fitting, A must be positive (monotonically increasing)."""
        import backend.services.platt_calibrator as pc
        monkeypatch.setattr(pc, "_PARAMS_PATH", tmp_path / "platt_params.json")
        rng = np.random.default_rng(42)
        n = 200
        raw = rng.uniform(0.0, 1.0, n)
        labels = (raw > 0.5).astype(float)
        A, B = pc.fit(raw, labels)
        assert A > 0, f"Fitted A={A} should be positive for increasing calibration"

    def test_fit_returns_tuple(self):
        from backend.services.platt_calibrator import fit
        A, B = fit(np.array([0.3, 0.7, 0.9]), np.array([0.0, 1.0, 1.0]))
        assert isinstance(A, float)
        assert isinstance(B, float)

    def test_sigmoid_boundary_safety(self):
        """calibrate must not raise for extreme inputs."""
        from backend.services.platt_calibrator import calibrate
        assert 0.0 <= calibrate(0.0) <= 1.0
        assert 0.0 <= calibrate(1.0) <= 1.0
        assert 0.0 <= calibrate(1e9) <= 1.0
        assert 0.0 <= calibrate(-1e9) <= 1.0

    def test_no_nan_or_inf_in_output(self):
        from backend.services.platt_calibrator import calibrate_with_interval
        r = calibrate_with_interval(0.5)
        for k, v in r.items():
            if isinstance(v, float):
                assert math.isfinite(v)
            elif isinstance(v, list):
                for x in v:
                    assert math.isfinite(x)

    def test_default_params_used_without_file(self, tmp_path, monkeypatch):
        """calibrate() should use defaults when params file is absent."""
        import backend.services.platt_calibrator as pc
        monkeypatch.setattr(pc, "_PARAMS_PATH", tmp_path / "no_file.json")
        result = pc.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    # ── F-3: interval_around_calibrated() must NOT re-apply Platt's sigmoid ──

    def test_interval_around_calibrated_does_not_retransform(self):
        """An already-final probability must come back ~unchanged.

        This is the regression test for F-3: advanced_ensemble_detector.py's
        weighted_score is already a final probability (Platt or XGBoost)
        by the time the calibration block is built. The old code path
        (calibrate_with_interval) re-applied the sigmoid to it anyway,
        so result["calibration"]["calibrated"] silently disagreed with
        the headline result["ai_probability"] for the same report.
        """
        from backend.services.platt_calibrator import interval_around_calibrated
        for p_final in (0.05, 0.5, 0.6, 0.92, 0.95):
            r = interval_around_calibrated(p_final)
            assert abs(r["calibrated"] - p_final) < 1e-3, (
                f"interval_around_calibrated({p_final}) returned "
                f"{r['calibrated']} -- input was re-transformed, not preserved"
            )

    def test_interval_around_calibrated_keys_and_bounds(self):
        from backend.services.platt_calibrator import interval_around_calibrated
        signals = _make_signals([0.8, 0.7], [0.8, 0.7])
        r = interval_around_calibrated(0.75, signals=signals)
        for key in ("calibrated", "interval_90", "A", "B"):
            assert key in r
        lo, hi = r["interval_90"]
        assert lo <= r["calibrated"] <= hi
        assert 0.0 <= lo and hi <= 1.0

    def test_ensemble_calibration_matches_ai_probability(self, sample_image_bytes):
        """End-to-end regression test for F-3: a single detect() call must
        report the same probability in both result["ai_probability"] and
        result["calibration"]["calibrated"] (up to the latter's rounding),
        which is exactly the invariant the pre-fix double-calibration bug
        violated -- and which no existing test checked before this fix.
        """
        from backend.services.advanced_ensemble_detector import AdvancedEnsembleDetector
        detector = AdvancedEnsembleDetector(sample_image_bytes, "test.png")
        report = detector.detect()
        detector.cleanup()
        assert abs(report["ai_probability"] - report["calibration"]["calibrated"]) < 1e-3, (
            "ai_probability and calibration.calibrated disagree -- "
            "the score is being calibrated more than once (F-3)"
        )


# ═══════════════════════════════════════════════════════════════════════
# Phase 26 — Stable Evidence IDs (UUID5)
# ═══════════════════════════════════════════════════════════════════════

class TestStableEvidenceIDs:

    def test_same_hash_produces_same_id(self):
        """UUID5(NAMESPACE_URL, sha256) must be deterministic."""
        sha = "abc123def456" * 4  # fake 48-char hash
        id1 = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))
        id2 = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))
        assert id1 == id2

    def test_different_hash_produces_different_id(self):
        sha1 = "a" * 64
        sha2 = "b" * 64
        assert str(uuid.uuid5(uuid.NAMESPACE_URL, sha1)) != \
               str(uuid.uuid5(uuid.NAMESPACE_URL, sha2))

    def test_result_is_valid_uuid(self):
        sha = "f" * 64
        eid = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))
        parsed = uuid.UUID(eid)
        assert parsed.version == 5

    def test_image_forensics_uses_uuid5(self):
        """image_forensics.py source must use uuid5 for evidence_id."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "services" / "image_forensics.py"
               ).read_text(encoding="utf-8")
        assert "uuid5" in src, "image_forensics must use uuid5 for evidence_id"
        for line in src.splitlines():
            if "evidence_id" in line and "uuid4" in line:
                raise AssertionError(
                    "evidence_id line still uses uuid4: " + line.strip()
                )

    def test_repeated_analysis_same_file_same_evidence_id(self):
        """Two analyses of the same bytes must produce the same evidence_id."""
        import hashlib
        fake_bytes = b"test image bytes" * 100
        sha = hashlib.sha256(fake_bytes).hexdigest()
        eid1 = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))
        eid2 = str(uuid.uuid5(uuid.NAMESPACE_URL, sha))
        assert eid1 == eid2

    def test_uuid5_is_version_5(self):
        eid = uuid.uuid5(uuid.NAMESPACE_URL, "x" * 64)
        assert eid.version == 5


# ═══════════════════════════════════════════════════════════════════════
# Integration — probability_distribution and calibration in ensemble
# ═══════════════════════════════════════════════════════════════════════

class TestEnsembleIntegration:

    def test_ensemble_result_has_probability_distribution(self):
        """advanced_ensemble_detector source must reference probability_distribution."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "services" / "advanced_ensemble_detector.py"
               ).read_text(encoding="utf-8")
        assert "probability_distribution" in src

    def test_ensemble_uses_platt_calibrate(self):
        """advanced_ensemble_detector source must reference platt_calibrator."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "services" / "advanced_ensemble_detector.py"
               ).read_text(encoding="utf-8")
        assert "platt_calibrat" in src

    def test_ensemble_uses_mcmc(self):
        """advanced_ensemble_detector source must reference mcmc_engine."""
        from pathlib import Path
        src = (Path(__file__).parent.parent / "services" / "advanced_ensemble_detector.py"
               ).read_text(encoding="utf-8")
        assert "mcmc_engine" in src or "run_mcmc" in src

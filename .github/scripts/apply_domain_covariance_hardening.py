"""Apply the reviewed domain-covariance hardening to the target checkout."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path("src/bayesian_phystwin/domain_covariance_calibration.py")
TESTS = Path(
    "tests/test_source_competence_domain_covariance_calibration_edges.py"
)


def _replace_record_validation(text: str) -> str:
    if "applied calibration requires all admissibility gates" in text:
        return text
    record_start = (
        '        for name in ("raw_covariance_sha256", '
        '"output_covariance_sha256"):\n'
    )
    record_end = (
        '            raise ValueError('
        '"exact fallback must preserve covariance identity")\n'
    )
    start = text.index(record_start)
    end = text.index(record_end, start) + len(record_end)
    record_lines = [
        '        for name in ("raw_covariance_sha256", '
        '"output_covariance_sha256"):',
        "            object.__setattr__(",
        "                self,",
        "                name,",
        "                sha256_digest(getattr(self, name), name=name),",
        "            )",
        "        if self.applied:",
        "            if not (",
        "                self.inference_admissible",
        "                and self.certificate_deployment_admissible",
        "                and self.calibration_supported",
        "            ):",
        "                raise ValueError(",
        '                    "applied calibration requires all '
        'admissibility gates"',
        "                )",
        "            if self.decision_id is None:",
        "                raise ValueError(",
        '                    "applied calibration requires a domain decision"',
        "                )",
        '            if self.reason != "calibration-domain-authorized":',
        "                raise ValueError(",
        '                    "applied calibration has invalid reason"',
        "                )",
        "            if (",
        "                self.covariance_scale == 1.0",
        "                and self.isotropic_variance == 0.0",
        "            ):",
        "                raise ValueError(",
        '                    "applied calibration must change the transform"',
        "                )",
        "        else:",
        "            if (",
        "                self.covariance_scale != 1.0",
        "                or self.isotropic_variance != 0.0",
        "            ):",
        "                raise ValueError(",
        '                    "fallback must use raw covariance transform"',
        "                )",
        "            if self.decision_id is None and self.calibration_supported:",
        "                raise ValueError(",
        '                    "unknown domain cannot be calibration-supported"',
        "                )",
        "            if not self.inference_admissible:",
        '                expected_reason = "inference-rejected"',
        "            elif self.decision_id is None:",
        '                expected_reason = "unknown-calibration-domain"',
        "            elif not self.certificate_deployment_admissible:",
        "                expected_reason = (",
        '                    "calibration-information-boundary-rejected"',
        "                )",
        "            elif not self.calibration_supported:",
        '                expected_reason = "calibration-domain-rejected"',
        "            else:",
        "                expected_reason = (",
        '                    "calibration-identity-transform-retained"',
        "                )",
        "            if self.reason != expected_reason:",
        "                raise ValueError(",
        '                    "fallback calibration has invalid reason"',
        "                )",
        "        if (",
        "            self.exact_fallback",
        "            and self.raw_covariance_sha256",
        "            != self.output_covariance_sha256",
        "        ):",
        "            raise ValueError(",
        '                "exact fallback must preserve covariance identity"',
        "            )",
    ]
    return text[:start] + "\n".join(record_lines) + "\n" + text[end:]


def _replace_application_routing(text: str) -> str:
    if "identity_transform = bool(" not in text:
        apply_start = '    decision = certificate.decision_for_domain(domain)\n'
        apply_boundary = "    if applied:\n"
        start = text.index(apply_start)
        end = text.index(apply_boundary, start)
        apply_lines = [
            "    decision = certificate.decision_for_domain(domain)",
            "    supported = bool(",
            "        decision is not None and decision.calibration_supported",
            "    )",
            "    identity_transform = bool(",
            "        decision is not None",
            "        and decision.selected_covariance_scale == 1.0",
            "        and decision.selected_isotropic_variance == 0.0",
            "    )",
            "    applied = (",
            "        inference_ok",
            "        and certificate.deployment_admissible",
            "        and supported",
            "        and not identity_transform",
            "    )",
            "    if not inference_ok:",
            '        reason = "inference-rejected"',
            "    elif decision is None:",
            '        reason = "unknown-calibration-domain"',
            "    elif not certificate.deployment_admissible:",
            '        reason = "calibration-information-boundary-rejected"',
            "    elif not decision.calibration_supported:",
            '        reason = "calibration-domain-rejected"',
            "    elif identity_transform:",
            '        reason = "calibration-identity-transform-retained"',
            "    else:",
            '        reason = "calibration-domain-authorized"',
            "    raw_digest = _array_digest(",
            "        np.asarray(raw_covariance, dtype=np.float64)",
            "    )",
        ]
        text = text[:start] + "\n".join(apply_lines) + "\n" + text[end:]
    old_digest = "        output_digest = _array_digest(transformed)\n"
    new_digest = "        output_digest = _array_digest(output)\n"
    if old_digest in text:
        text = text.replace(old_digest, new_digest, 1)
    elif new_digest not in text:
        raise RuntimeError("output digest patch target changed")
    return text


def _extend_adversarial_tests(text: str) -> str:
    applied_marker = (
        '        lambda: replace(applied, reason="wrong", artifact_id=None),\n'
    )
    applied_addition = (
        "        lambda: replace(\n"
        "            applied,\n"
        "            decision_id=None,\n"
        "            artifact_id=None,\n"
        "        ),\n"
    )
    if applied_addition not in text:
        index = text.index(applied_marker) + len(applied_marker)
        text = text[:index] + applied_addition + text[index:]

    fallback_marker = (
        '        lambda: replace(fallback, reason="wrong", artifact_id=None),\n'
    )
    fallback_addition = (
        "        lambda: replace(\n"
        "            fallback,\n"
        '            output_covariance_sha256="0" * 64,\n'
        "            artifact_id=None,\n"
        "        ),\n"
    )
    if fallback_addition not in text:
        index = text.index(fallback_marker) + len(fallback_marker)
        text = text[:index] + fallback_addition + text[index:]
    return text


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    source_text = _replace_record_validation(source_text)
    source_text = _replace_application_routing(source_text)
    SOURCE.write_text(source_text, encoding="utf-8")

    test_text = TESTS.read_text(encoding="utf-8")
    TESTS.write_text(_extend_adversarial_tests(test_text), encoding="utf-8")


if __name__ == "__main__":
    main()

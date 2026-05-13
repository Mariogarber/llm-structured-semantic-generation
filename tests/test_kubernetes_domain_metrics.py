from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.auxiliary_text_metrics import compute_auxiliary_text_metrics
from llm_structured_semantic_generation.evaluation import evaluate_yaml_prediction, summarize_evaluations


HARDENED_DEPLOYMENT_WITH_SERVICE = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 2
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx:1.25
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 256Mi
        securityContext:
          runAsNonRoot: true
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
---
apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 80
"""


MINIMAL_POD_WITH_SMELLS = """\
apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  containers:
  - name: web
    image: nginx:latest
"""


class KubernetesDomainMetricsTest(unittest.TestCase):
    def test_hardened_multiresource_manifest_reaches_level_5(self) -> None:
        evaluation = evaluate_yaml_prediction(
            HARDENED_DEPLOYMENT_WITH_SERVICE,
            HARDENED_DEPLOYMENT_WITH_SERVICE,
        )
        summary = summarize_evaluations([evaluation])

        self.assertTrue(evaluation.yaml_parse_ok)
        self.assertEqual(evaluation.kubernetes_domain_validity_level, 5)
        self.assertTrue(evaluation.kubernetes_domain_gate_pass)
        self.assertEqual(evaluation.kubernetes_domain_validity_score, 1.0)
        self.assertEqual(evaluation.kubernetes_domain_errors, ())
        self.assertEqual(summary["kubernetes_domain_gate_pass_rate"], 1.0)
        self.assertEqual(summary["kubernetes_level_5_pass_rate"], 1.0)

    def test_minimal_pod_passes_identity_but_fails_static_smells(self) -> None:
        evaluation = evaluate_yaml_prediction(MINIMAL_POD_WITH_SMELLS, MINIMAL_POD_WITH_SMELLS)
        categories = {error["category"] for error in evaluation.kubernetes_domain_errors}

        self.assertEqual(evaluation.kubernetes_domain_validity_level, 4)
        self.assertFalse(evaluation.kubernetes_domain_gate_pass)
        self.assertIn("missing_resource_requirement", categories)
        self.assertIn("latest_image_tag", categories)
        self.assertIn("missing_run_as_non_root", categories)

    def test_selector_mismatch_blocks_level_3(self) -> None:
        predicted = HARDENED_DEPLOYMENT_WITH_SERVICE.replace("app: web", "app: api", 1)

        evaluation = evaluate_yaml_prediction(HARDENED_DEPLOYMENT_WITH_SERVICE, predicted)
        categories = {error["category"] for error in evaluation.kubernetes_domain_errors}

        self.assertEqual(evaluation.kubernetes_domain_validity_level, 2)
        self.assertIn("selector_template_mismatch", categories)

    def test_service_without_matching_workload_blocks_level_4(self) -> None:
        predicted = HARDENED_DEPLOYMENT_WITH_SERVICE.replace(
            "  selector:\n    app: web\n  ports:",
            "  selector:\n    app: other\n  ports:",
        )

        evaluation = evaluate_yaml_prediction(HARDENED_DEPLOYMENT_WITH_SERVICE, predicted)
        categories = {error["category"] for error in evaluation.kubernetes_domain_errors}

        self.assertLessEqual(evaluation.kubernetes_domain_validity_level, 3)
        self.assertIn("service_selector_without_workload", categories)

    def test_auxiliary_text_metrics_identical_partial_and_empty(self) -> None:
        identical = compute_auxiliary_text_metrics(MINIMAL_POD_WITH_SMELLS, MINIMAL_POD_WITH_SMELLS)
        partial = compute_auxiliary_text_metrics(
            MINIMAL_POD_WITH_SMELLS,
            MINIMAL_POD_WITH_SMELLS.replace("nginx:latest", "redis:7"),
        )
        empty = compute_auxiliary_text_metrics(MINIMAL_POD_WITH_SMELLS, "")

        self.assertEqual(identical["bleu_score"], 1.0)
        self.assertEqual(identical["rouge1_f1"], 1.0)
        self.assertEqual(identical["rouge2_f1"], 1.0)
        self.assertEqual(identical["rougeL_f1"], 1.0)
        self.assertIsNone(identical["perplexity"])
        self.assertFalse(identical["perplexity_available"])
        self.assertGreater(partial["bleu_score"], 0.0)
        self.assertLess(partial["bleu_score"], 1.0)
        self.assertEqual(empty["bleu_score"], 0.0)

    def test_perplexity_signal_is_preserved_when_available(self) -> None:
        evaluation = evaluate_yaml_prediction(
            MINIMAL_POD_WITH_SMELLS,
            MINIMAL_POD_WITH_SMELLS,
            perplexity=12.5,
        )
        summary = summarize_evaluations([evaluation])

        self.assertEqual(evaluation.auxiliary_text_metrics["perplexity"], 12.5)
        self.assertTrue(evaluation.auxiliary_text_metrics["perplexity_available"])
        self.assertEqual(summary["average_perplexity"], 12.5)
        self.assertEqual(summary["perplexity_available_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

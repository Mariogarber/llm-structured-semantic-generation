from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_structured_semantic_generation.evaluation import evaluate_yaml_prediction
from llm_structured_semantic_generation.prompt_requirements import (
    evaluate_prompt_requirements,
    evaluate_required_fields,
    extract_prompt_requirements,
    extract_yaml_requirement_atoms,
)
from llm_structured_semantic_generation.structure import parse_yaml_documents


class PromptRequirementsTest(unittest.TestCase):
    def test_extract_prompt_requirements_captures_kind_name_image_and_label(self) -> None:
        prompt_text = (
            'Please write a k8s YAML file to create a deployment named "redis-app" '
            'that uses image "docker.io/redis:6.0.5". The pods should be labeled with "app: redis".'
        )

        atoms = extract_prompt_requirements(prompt_text)
        canonicals = {atom.canonical for atom in atoms}

        self.assertIn("kind=Deployment", canonicals)
        self.assertIn("metadata.name=redis-app", canonicals)
        self.assertIn("image=docker.io/redis:6.0.5", canonicals)
        self.assertIn("label:app=redis", canonicals)

    def test_extract_yaml_requirement_atoms_reads_common_runtime_facts(self) -> None:
        yaml_text = (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: redis-app\n"
            "spec:\n"
            "  replicas: 3\n"
            "  selector:\n"
            "    matchLabels:\n"
            "      app: redis\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        app: redis\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: redis\n"
            "        image: docker.io/redis:6.0.5\n"
            "        ports:\n"
            "        - containerPort: 6379\n"
        )

        atoms = extract_yaml_requirement_atoms(yaml_text)
        canonicals = {atom.canonical for atom in atoms}

        self.assertIn("kind=Deployment", canonicals)
        self.assertIn("metadata.name=redis-app", canonicals)
        self.assertIn("replicas=3", canonicals)
        self.assertIn("image=docker.io/redis:6.0.5", canonicals)
        self.assertIn("label:app=redis", canonicals)
        self.assertIn("port=6379", canonicals)

    def test_prompt_requirement_evaluation_matches_reference_yaml(self) -> None:
        prompt_text = (
            'Create a Deployment called custom-deploy. Use the nginx:latest image, '
            'have 1 replica, and expose port 80.'
        )
        yaml_text = (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: custom-deploy\n"
            "spec:\n"
            "  replicas: 1\n"
            "  selector:\n"
            "    matchLabels:\n"
            "      app: demo\n"
            "  template:\n"
            "    metadata:\n"
            "      labels:\n"
            "        app: demo\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: nginx\n"
            "        image: nginx:latest\n"
            "        ports:\n"
            "        - containerPort: 80\n"
        )

        evaluation = evaluate_prompt_requirements(prompt_text, parse_yaml_documents(yaml_text))

        self.assertTrue(evaluation.prompt_requirement_supported)
        self.assertEqual(evaluation.prompt_requirement_count, 5)
        self.assertEqual(evaluation.prompt_requirement_precision, 1.0)
        self.assertEqual(evaluation.prompt_requirement_recall, 1.0)
        self.assertEqual(evaluation.prompt_requirement_f1, 1.0)
        self.assertTrue(evaluation.prompt_requirement_exact_match)

    def test_required_fields_detect_missing_deployment_selector(self) -> None:
        yaml_text = (
            "apiVersion: apps/v1\n"
            "kind: Deployment\n"
            "metadata:\n"
            "  name: broken-deploy\n"
            "spec:\n"
            "  template:\n"
            "    spec:\n"
            "      containers:\n"
            "      - name: nginx\n"
            "        image: nginx:latest\n"
        )

        evaluation = evaluate_required_fields(parse_yaml_documents(yaml_text))

        self.assertEqual(evaluation.required_field_applicable_resource_count, 1)
        self.assertIsNotNone(evaluation.required_field_presence_rate)
        self.assertLess(evaluation.required_field_presence_rate, 1.0)
        self.assertEqual(evaluation.required_field_complete_resource_rate, 0.0)
        self.assertFalse(evaluation.required_field_complete_sample)

    def test_yaml_evaluation_exposes_new_prompt_and_required_field_metrics(self) -> None:
        prompt_text = 'Create a ConfigMap named database-config with key-values DB_HOST=database-host and DB_PORT=5432.'
        yaml_text = (
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "metadata:\n"
            "  name: database-config\n"
            "data:\n"
            "  DB_HOST: database-host\n"
            "  DB_PORT: '5432'\n"
        )

        evaluation = evaluate_yaml_prediction(yaml_text, yaml_text, prompt_text=prompt_text)

        self.assertTrue(evaluation.prompt_requirement_supported)
        self.assertEqual(evaluation.prompt_requirement_precision, 1.0)
        self.assertEqual(evaluation.prompt_requirement_recall, 1.0)
        self.assertEqual(evaluation.required_field_complete_resource_rate, 1.0)
        self.assertTrue(evaluation.required_field_complete_sample)


if __name__ == "__main__":
    unittest.main()

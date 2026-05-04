from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .structure import parse_yaml_documents


KNOWN_KINDS = (
    "CustomResourceDefinition",
    "HorizontalPodAutoscaler",
    "PersistentVolumeClaim",
    "PodDisruptionBudget",
    "PersistentVolume",
    "ServiceAccount",
    "NetworkPolicy",
    "StatefulSet",
    "ReplicaSet",
    "RoleBinding",
    "DaemonSet",
    "ConfigMap",
    "LimitRange",
    "CronJob",
    "Deployment",
    "Endpoints",
    "Ingress",
    "Service",
    "Secret",
    "Pod",
    "Job",
    "Role",
)

KIND_BY_LOWER = {kind.lower(): kind for kind in KNOWN_KINDS}
KIND_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(kind) for kind in sorted(KNOWN_KINDS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

NAME_PATTERNS = (
    re.compile(r"\bwith the name\s+[\"']?([A-Za-z0-9._-]+)[\"']?", flags=re.IGNORECASE),
    re.compile(r"\bname of the [a-z0-9 _-]+ has to be\s+[\"']?([A-Za-z0-9._-]+)[\"']?", flags=re.IGNORECASE),
    re.compile(r"\b(?:named|called)\s+[\"']?([A-Za-z0-9._-]+)[\"']?", flags=re.IGNORECASE),
)

IMAGE_PATTERNS = (
    re.compile(r"\bimage\s*[\"']([^\"']+)[\"']", flags=re.IGNORECASE),
    re.compile(r"\b(?:use|using|with)\s+(?:the\s+)?([A-Za-z0-9./_-]+:[A-Za-z0-9._-]+)\s+image\b", flags=re.IGNORECASE),
    re.compile(r"\b(?:use|using)\s+(?:the\s+)?latest\s+([A-Za-z0-9./_-]+)\s+image\b", flags=re.IGNORECASE),
)

LABEL_PATTERNS = (
    re.compile(r"\blabel(?:ed|s)?\s+(?:with\s+)?[\"']?([A-Za-z0-9_.-]+)\s*[:=]\s*([A-Za-z0-9_.-]+)[\"']?", flags=re.IGNORECASE),
    re.compile(r"\bpods?\s+labeled\s+[\"']?([A-Za-z0-9_.-]+)\s*[:=]\s*([A-Za-z0-9_.-]+)[\"']?", flags=re.IGNORECASE),
)

ENV_ASSIGNMENT_PATTERN = re.compile(r"\b([A-Z][A-Z0-9_]+)\s*=\s*([A-Za-z0-9._:/-]+)\b")
ENV_SHARED_VALUE_PATTERN = re.compile(
    r"\b([A-Z][A-Z0-9_]+)\s+and\s+([A-Z][A-Z0-9_]+),?\s+both with (?:the\s+)?value\s+([A-Za-z0-9._:/-]+)\b"
)
REPLICA_PATTERN = re.compile(r"\b(\d+)\s+replicas?\b", flags=re.IGNORECASE)
AUTOSCALING_RANGE_PATTERN = re.compile(
    r"\breplicas?.{0,40}?\bbetween\s+(\d+)\s+and\s+(\d+)\b",
    flags=re.IGNORECASE,
)
PORT_PATTERN = re.compile(
    r"\b(?:expose\s+port|mapped to port|port(?:s)?(?:\s+is|\s+should be|:|=| to| on)?)\s*(\d{2,5})\b",
    flags=re.IGNORECASE,
)
TARGET_PORT_PATTERN = re.compile(r"\btargetport\s*[:=]?\s*(\d{2,5})\b", flags=re.IGNORECASE)
NAMESPACE_PATTERN = re.compile(r"\bnamespace\s+[\"']?([A-Za-z0-9._-]+)[\"']?", flags=re.IGNORECASE)
SERVICE_ACCOUNT_PATTERN = re.compile(
    r"\bservice account(?:\s+with the name)?\s+[\"']?([A-Za-z0-9._-]+)[\"']?",
    flags=re.IGNORECASE,
)
CRON_EVERY_MINUTE_PATTERN = re.compile(r"\bevery minute\b", flags=re.IGNORECASE)
CRON_EVERY_N_MINUTES_PATTERN = re.compile(r"\bevery\s+(\d+)\s+minutes?\b", flags=re.IGNORECASE)
STORAGE_CAPACITY_PATTERN = re.compile(r"\bcapacity of\s+(\d+(?:Mi|Gi|Ti))\b", flags=re.IGNORECASE)
STORAGE_MIN_MAX_PATTERN = re.compile(
    r"\bminimum and maximum storage limits?.{0,60}?\b(\d+(?:Mi|Gi|Ti))\s+and\s+(\d+(?:Mi|Gi|Ti))\b",
    flags=re.IGNORECASE,
)
HOST_PATH_PATTERN = re.compile(r"\bdirectory at\s+[\"']([^\"']+)[\"']", flags=re.IGNORECASE)


RequiredPathGroup = tuple[tuple[str, ...], ...]

GENERIC_REQUIRED_FIELD_GROUPS: tuple[RequiredPathGroup, ...] = (
    (("apiVersion",),),
    (("kind",),),
    (("metadata", "name"),),
)

KIND_REQUIRED_FIELD_GROUPS: dict[str, tuple[RequiredPathGroup, ...]] = {
    "Pod": (
        (("spec", "containers"),),
    ),
    "Deployment": (
        (("spec", "selector"),),
        (("spec", "template"),),
        (("spec", "template", "spec", "containers"),),
    ),
    "DaemonSet": (
        (("spec", "selector"),),
        (("spec", "template"),),
        (("spec", "template", "spec", "containers"),),
    ),
    "StatefulSet": (
        (("spec", "serviceName"),),
        (("spec", "selector"),),
        (("spec", "template"),),
        (("spec", "template", "spec", "containers"),),
    ),
    "ReplicaSet": (
        (("spec", "selector"),),
        (("spec", "template"),),
        (("spec", "template", "spec", "containers"),),
    ),
    "CronJob": (
        (("spec", "schedule"),),
        (("spec", "jobTemplate", "spec", "template", "spec", "containers"),),
    ),
    "Job": (
        (("spec", "template", "spec", "containers"),),
    ),
    "Service": (
        (("spec", "ports"),),
    ),
    "PersistentVolumeClaim": (
        (("spec", "resources", "requests", "storage"),),
    ),
    "PersistentVolume": (
        (("spec", "capacity", "storage"),),
    ),
    "Ingress": (
        (("spec", "rules"),),
    ),
    "Endpoints": (
        (("subsets",),),
    ),
    "HorizontalPodAutoscaler": (
        (("spec", "scaleTargetRef"),),
        (("spec", "maxReplicas"),),
    ),
    "NetworkPolicy": (
        (("spec", "podSelector"),),
    ),
    "PodDisruptionBudget": (
        (("spec", "maxUnavailable"), ("spec", "minAvailable")),
    ),
    "LimitRange": (
        (("spec", "limits"),),
    ),
    "RoleBinding": (
        (("roleRef",),),
        (("subjects",),),
    ),
    "CustomResourceDefinition": (
        (("spec", "group"),),
        (("spec", "names", "plural"),),
        (("spec", "names", "kind"),),
        (("spec", "versions"),),
    ),
}


@dataclass(frozen=True)
class RequirementAtom:
    category: str
    value: str
    key: str | None = None

    @property
    def canonical(self) -> str:
        if self.key:
            return f"{self.category}:{self.key}={self.value}"
        return f"{self.category}={self.value}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["canonical"] = self.canonical
        return payload


@dataclass(frozen=True)
class PromptRequirementEvaluation:
    prompt_requirement_supported: bool
    prompt_requirement_count: int
    comparable_prediction_requirement_count: int
    matched_prompt_requirement_count: int
    prompt_requirement_precision: float | None
    prompt_requirement_recall: float | None
    prompt_requirement_f1: float | None
    prompt_requirement_exact_match: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequiredFieldEvaluation:
    required_field_applicable_resource_count: int
    required_field_total_count: int
    required_field_present_count: int
    required_field_presence_rate: float | None
    required_field_complete_resource_rate: float | None
    required_field_complete_sample: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text_value(value: str) -> str:
    return value.strip().strip("\"'").strip()


def _normalize_identifier_value(value: str) -> str:
    return _normalize_text_value(value).rstrip(".,;")


def _normalize_kind(kind: str) -> str:
    return KIND_BY_LOWER.get(kind.lower(), kind)


def _dedupe_atoms(atoms: Iterable[RequirementAtom]) -> tuple[RequirementAtom, ...]:
    seen: dict[str, RequirementAtom] = {}
    for atom in atoms:
        seen.setdefault(atom.canonical, atom)
    return tuple(seen.values())


def _mapping_at(value: Any, *path: str) -> dict[str, Any] | None:
    current = value
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current if isinstance(current, dict) else None


def _value_at(value: Any, *path: str) -> Any:
    current = value
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _path_present(document: Any, path: tuple[str, ...]) -> bool:
    current = _value_at(document, *path)
    if current is None:
        return False
    if isinstance(current, (list, dict, tuple, set)):
        return len(current) > 0
    return True


def _pod_specs(document: Any) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if not isinstance(document, dict):
        return specs

    root_spec = document.get("spec")
    if isinstance(root_spec, dict):
        if any(key in root_spec for key in ("containers", "initContainers", "ephemeralContainers", "volumes", "serviceAccountName")):
            specs.append(root_spec)
        template_spec = _mapping_at(root_spec, "template", "spec")
        if template_spec is not None:
            specs.append(template_spec)
        job_template_spec = _mapping_at(root_spec, "jobTemplate", "spec", "template", "spec")
        if job_template_spec is not None:
            specs.append(job_template_spec)
    return specs


def _container_lists(pod_spec: dict[str, Any]) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    for key in ("containers", "initContainers", "ephemeralContainers"):
        value = pod_spec.get(key)
        if isinstance(value, list):
            containers.extend(item for item in value if isinstance(item, dict))
    return containers


def _label_mappings(document: Any) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for path in (
        ("metadata", "labels"),
        ("spec", "selector", "matchLabels"),
        ("spec", "template", "metadata", "labels"),
        ("spec", "podSelector", "matchLabels"),
    ):
        mapping = _mapping_at(document, *path)
        if mapping is not None:
            mappings.append(mapping)
    return mappings


def _set_precision_recall_f1(
    reference: set[str],
    prediction: set[str],
) -> tuple[float, float, float]:
    if not reference:
        return 0.0, 0.0, 0.0
    matched = len(reference & prediction)
    precision = matched / len(prediction) if prediction else 0.0
    recall = matched / len(reference)
    f1 = (2 * precision * recall / (precision + recall)) if (precision or recall) else 0.0
    return precision, recall, f1


def _assignment_category(prompt_text: str, match_start: int) -> str:
    lower_prompt = prompt_text.lower()
    context = lower_prompt[max(0, match_start - 60) : match_start + 40]
    if any(token in context for token in ("environment variable", "environment variables", " env ", " env-", " env_")):
        return "env"
    if "configmap" in lower_prompt or "key-values" in context or "key values" in context:
        return "data"
    return "env"


def _filter_prompt_kinds(prompt_text: str, detected_kinds: list[str]) -> list[str]:
    kinds = list(dict.fromkeys(detected_kinds))
    lower_prompt = prompt_text.lower()
    filtered = set(kinds)

    if "CronJob" in filtered and "Job" in filtered:
        filtered.remove("Job")

    workload_kinds = {"Deployment", "DaemonSet", "StatefulSet", "ReplicaSet", "Job", "CronJob"}
    explicit_pod_request = bool(
        re.search(r"\b(create|define|write).{0,20}\bpod\b", lower_prompt)
        or re.search(r"\bpod yaml\b", lower_prompt)
        or re.search(r"\bkind\s*:\s*pod\b", lower_prompt)
    )
    if "Pod" in filtered and (filtered & workload_kinds) and not explicit_pod_request:
        filtered.remove("Pod")

    return [kind for kind in kinds if kind in filtered]


def extract_prompt_requirements(prompt_text: str) -> tuple[RequirementAtom, ...]:
    atoms: list[RequirementAtom] = []
    prompt = prompt_text or ""
    lower_prompt = prompt.lower()

    detected_kinds = [_normalize_kind(match.group(1)) for match in KIND_PATTERN.finditer(prompt)]
    for kind in _filter_prompt_kinds(prompt, detected_kinds):
        atoms.append(RequirementAtom(category="kind", value=kind))

    if "default namespace" in lower_prompt:
        atoms.append(RequirementAtom(category="namespace", value="default"))
    for match in NAMESPACE_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="namespace", value=_normalize_text_value(match.group(1))))

    for pattern in NAME_PATTERNS:
        for match in pattern.finditer(prompt):
            atoms.append(RequirementAtom(category="metadata.name", value=_normalize_identifier_value(match.group(1))))

    for pattern in IMAGE_PATTERNS:
        for match in pattern.finditer(prompt):
            image = _normalize_text_value(match.group(1))
            if pattern is IMAGE_PATTERNS[2] and ":" not in image:
                image = f"{image}:latest"
            atoms.append(RequirementAtom(category="image", value=image))

    for match in REPLICA_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="replicas", value=match.group(1)))
    for match in AUTOSCALING_RANGE_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="autoscaling.minReplicas", value=match.group(1)))
        atoms.append(RequirementAtom(category="autoscaling.maxReplicas", value=match.group(2)))

    for match in PORT_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="port", value=match.group(1)))
    for match in TARGET_PORT_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="targetPort", value=match.group(1)))

    for match in ENV_ASSIGNMENT_PATTERN.finditer(prompt):
        category = _assignment_category(prompt, match.start())
        atoms.append(
            RequirementAtom(
                category=category,
                key=match.group(1),
                value=_normalize_text_value(match.group(2)),
            )
        )
    for match in ENV_SHARED_VALUE_PATTERN.finditer(prompt):
        shared_value = _normalize_text_value(match.group(3))
        category = _assignment_category(prompt, match.start())
        atoms.append(RequirementAtom(category=category, key=match.group(1), value=shared_value))
        atoms.append(RequirementAtom(category=category, key=match.group(2), value=shared_value))

    for pattern in LABEL_PATTERNS:
        for match in pattern.finditer(prompt):
            atoms.append(
                RequirementAtom(
                    category="label",
                    key=_normalize_text_value(match.group(1)),
                    value=_normalize_text_value(match.group(2)),
                )
            )

    for match in SERVICE_ACCOUNT_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="serviceAccountName", value=_normalize_text_value(match.group(1))))

    if CRON_EVERY_MINUTE_PATTERN.search(prompt):
        atoms.append(RequirementAtom(category="cron.schedule", value="* * * * *"))
    for match in CRON_EVERY_N_MINUTES_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="cron.schedule", value=f"*/{match.group(1)} * * * *"))

    for match in STORAGE_CAPACITY_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="storage.request", value=match.group(1)))
    for match in STORAGE_MIN_MAX_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="storage.min", value=match.group(1)))
        atoms.append(RequirementAtom(category="storage.max", value=match.group(2)))

    for match in HOST_PATH_PATTERN.finditer(prompt):
        atoms.append(RequirementAtom(category="hostPath.path", value=_normalize_text_value(match.group(1))))

    return _dedupe_atoms(atoms)


def extract_yaml_requirement_atoms_from_documents(documents: tuple[Any, ...]) -> tuple[RequirementAtom, ...]:
    atoms: list[RequirementAtom] = []
    for document in documents:
        if not isinstance(document, dict):
            continue

        kind = document.get("kind")
        if isinstance(kind, str):
            atoms.append(RequirementAtom(category="kind", value=kind))

        metadata = document.get("metadata")
        if isinstance(metadata, dict):
            name = metadata.get("name")
            if isinstance(name, str):
                atoms.append(RequirementAtom(category="metadata.name", value=name))
            namespace = metadata.get("namespace")
            if isinstance(namespace, str):
                atoms.append(RequirementAtom(category="namespace", value=namespace))

        replicas = _value_at(document, "spec", "replicas")
        if isinstance(replicas, int):
            atoms.append(RequirementAtom(category="replicas", value=str(replicas)))

        min_replicas = _value_at(document, "spec", "minReplicas")
        if isinstance(min_replicas, int):
            atoms.append(RequirementAtom(category="autoscaling.minReplicas", value=str(min_replicas)))
        max_replicas = _value_at(document, "spec", "maxReplicas")
        if isinstance(max_replicas, int):
            atoms.append(RequirementAtom(category="autoscaling.maxReplicas", value=str(max_replicas)))

        schedule = _value_at(document, "spec", "schedule")
        if isinstance(schedule, str):
            atoms.append(RequirementAtom(category="cron.schedule", value=schedule))

        pvc_storage = _value_at(document, "spec", "resources", "requests", "storage")
        if isinstance(pvc_storage, str):
            atoms.append(RequirementAtom(category="storage.request", value=pvc_storage))

        pv_storage = _value_at(document, "spec", "capacity", "storage")
        if isinstance(pv_storage, str):
            atoms.append(RequirementAtom(category="storage.request", value=pv_storage))

        host_path = _value_at(document, "spec", "hostPath", "path")
        if isinstance(host_path, str):
            atoms.append(RequirementAtom(category="hostPath.path", value=host_path))

        if kind == "ConfigMap":
            data = document.get("data")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(key, str) and value is not None:
                        atoms.append(RequirementAtom(category="data", key=key, value=str(value)))

        limits = _value_at(document, "spec", "limits")
        if isinstance(limits, list):
            for limit in limits:
                if not isinstance(limit, dict):
                    continue
                minimum_storage = _value_at(limit, "min", "storage")
                if isinstance(minimum_storage, str):
                    atoms.append(RequirementAtom(category="storage.min", value=minimum_storage))
                maximum_storage = _value_at(limit, "max", "storage")
                if isinstance(maximum_storage, str):
                    atoms.append(RequirementAtom(category="storage.max", value=maximum_storage))

        ports = _value_at(document, "spec", "ports")
        if isinstance(ports, list):
            for port_row in ports:
                if not isinstance(port_row, dict):
                    continue
                port = port_row.get("port")
                if isinstance(port, int):
                    atoms.append(RequirementAtom(category="port", value=str(port)))
                target_port = port_row.get("targetPort")
                if isinstance(target_port, (int, str)):
                    atoms.append(RequirementAtom(category="targetPort", value=str(target_port)))

        for label_mapping in _label_mappings(document):
            for key, value in label_mapping.items():
                if isinstance(key, str) and isinstance(value, str):
                    atoms.append(RequirementAtom(category="label", key=key, value=value))

        for pod_spec in _pod_specs(document):
            service_account_name = pod_spec.get("serviceAccountName")
            if isinstance(service_account_name, str):
                atoms.append(RequirementAtom(category="serviceAccountName", value=service_account_name))

            for container in _container_lists(pod_spec):
                image = container.get("image")
                if isinstance(image, str):
                    atoms.append(RequirementAtom(category="image", value=image))

                env_rows = container.get("env")
                if isinstance(env_rows, list):
                    for env_row in env_rows:
                        if not isinstance(env_row, dict):
                            continue
                        name = env_row.get("name")
                        value = env_row.get("value")
                        if isinstance(name, str) and isinstance(value, str):
                            atoms.append(RequirementAtom(category="env", key=name, value=value))

                port_rows = container.get("ports")
                if isinstance(port_rows, list):
                    for port_row in port_rows:
                        if not isinstance(port_row, dict):
                            continue
                        port = port_row.get("containerPort")
                        if isinstance(port, int):
                            atoms.append(RequirementAtom(category="port", value=str(port)))

    return _dedupe_atoms(atoms)


def extract_yaml_requirement_atoms(yaml_text: str) -> tuple[RequirementAtom, ...]:
    return extract_yaml_requirement_atoms_from_documents(parse_yaml_documents(yaml_text))


def evaluate_prompt_requirements(
    prompt_text: str | None,
    prediction_documents: tuple[Any, ...],
) -> PromptRequirementEvaluation:
    if not prompt_text:
        return PromptRequirementEvaluation(
            prompt_requirement_supported=False,
            prompt_requirement_count=0,
            comparable_prediction_requirement_count=0,
            matched_prompt_requirement_count=0,
            prompt_requirement_precision=None,
            prompt_requirement_recall=None,
            prompt_requirement_f1=None,
            prompt_requirement_exact_match=None,
        )

    prompt_atoms = extract_prompt_requirements(prompt_text)
    if not prompt_atoms:
        return PromptRequirementEvaluation(
            prompt_requirement_supported=False,
            prompt_requirement_count=0,
            comparable_prediction_requirement_count=0,
            matched_prompt_requirement_count=0,
            prompt_requirement_precision=None,
            prompt_requirement_recall=None,
            prompt_requirement_f1=None,
            prompt_requirement_exact_match=None,
        )

    predicted_atoms = extract_yaml_requirement_atoms_from_documents(prediction_documents)
    prompt_categories = {atom.category for atom in prompt_atoms}
    prompt_atom_keys = {atom.canonical for atom in prompt_atoms}
    comparable_prediction_atom_keys = {
        atom.canonical for atom in predicted_atoms if atom.category in prompt_categories
    }
    matched_atom_keys = prompt_atom_keys & comparable_prediction_atom_keys
    precision, recall, f1 = _set_precision_recall_f1(prompt_atom_keys, comparable_prediction_atom_keys)

    return PromptRequirementEvaluation(
        prompt_requirement_supported=True,
        prompt_requirement_count=len(prompt_atom_keys),
        comparable_prediction_requirement_count=len(comparable_prediction_atom_keys),
        matched_prompt_requirement_count=len(matched_atom_keys),
        prompt_requirement_precision=precision,
        prompt_requirement_recall=recall,
        prompt_requirement_f1=f1,
        prompt_requirement_exact_match=prompt_atom_keys == comparable_prediction_atom_keys,
    )


def evaluate_required_fields(prediction_documents: tuple[Any, ...]) -> RequiredFieldEvaluation:
    applicable_resource_count = 0
    required_field_total_count = 0
    required_field_present_count = 0
    complete_resource_count = 0

    for document in prediction_documents:
        if not isinstance(document, dict):
            continue
        kind = document.get("kind")
        if not isinstance(kind, str):
            continue

        applicable_resource_count += 1
        required_groups = GENERIC_REQUIRED_FIELD_GROUPS + KIND_REQUIRED_FIELD_GROUPS.get(kind, ())
        required_field_total_count += len(required_groups)

        present_groups = 0
        for group in required_groups:
            if any(_path_present(document, path) for path in group):
                present_groups += 1
        required_field_present_count += present_groups
        if present_groups == len(required_groups):
            complete_resource_count += 1

    if applicable_resource_count == 0 or required_field_total_count == 0:
        return RequiredFieldEvaluation(
            required_field_applicable_resource_count=applicable_resource_count,
            required_field_total_count=required_field_total_count,
            required_field_present_count=required_field_present_count,
            required_field_presence_rate=None,
            required_field_complete_resource_rate=None,
            required_field_complete_sample=None,
        )

    return RequiredFieldEvaluation(
        required_field_applicable_resource_count=applicable_resource_count,
        required_field_total_count=required_field_total_count,
        required_field_present_count=required_field_present_count,
        required_field_presence_rate=required_field_present_count / required_field_total_count,
        required_field_complete_resource_rate=complete_resource_count / applicable_resource_count,
        required_field_complete_sample=complete_resource_count == applicable_resource_count,
    )


def summarize_prompt_requirement_atoms(rows: Iterable[tuple[RequirementAtom, ...]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for atoms in rows:
        for atom in atoms:
            counter[atom.category] += 1
    return dict(counter)

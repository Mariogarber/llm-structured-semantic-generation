from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .prompt_requirements import GENERIC_REQUIRED_FIELD_GROUPS, KIND_REQUIRED_FIELD_GROUPS, KNOWN_KINDS


WORKLOAD_KINDS = {"Deployment", "DaemonSet", "StatefulSet", "ReplicaSet"}
POD_TEMPLATE_WORKLOAD_KINDS = WORKLOAD_KINDS | {"Job", "CronJob"}


@dataclass(frozen=True)
class KubernetesDomainEvaluation:
    kubernetes_domain_validity_level: int
    kubernetes_domain_gate_pass: bool
    kubernetes_domain_validity_score: float
    kubernetes_domain_level_scores: dict[str, float | None]
    kubernetes_domain_errors: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _error(level: int, category: str, message: str, *, path: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"level": level, "category": category, "message": message}
    if path:
        payload["path"] = path
    return payload


def _value_at(value: Any, *path: str) -> Any:
    current = value
    for segment in path:
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def _mapping_at(value: Any, *path: str) -> dict[str, Any] | None:
    current = _value_at(value, *path)
    return current if isinstance(current, dict) else None


def _path_present(document: Any, path: tuple[str, ...]) -> bool:
    current = _value_at(document, *path)
    if current is None:
        return False
    if isinstance(current, (dict, list, tuple, set)):
        return len(current) > 0
    return True


def _pod_specs(document: Any) -> list[tuple[str, dict[str, Any]]]:
    specs: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(document, dict):
        return specs

    root_spec = document.get("spec")
    if isinstance(root_spec, dict):
        if any(key in root_spec for key in ("containers", "initContainers", "ephemeralContainers", "volumes", "serviceAccountName")):
            specs.append(("spec", root_spec))
        template_spec = _mapping_at(root_spec, "template", "spec")
        if template_spec is not None:
            specs.append(("spec.template.spec", template_spec))
        job_template_spec = _mapping_at(root_spec, "jobTemplate", "spec", "template", "spec")
        if job_template_spec is not None:
            specs.append(("spec.jobTemplate.spec.template.spec", job_template_spec))
    return specs


def _container_lists(pod_spec: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    containers: list[tuple[str, dict[str, Any]]] = []
    for key in ("containers", "initContainers", "ephemeralContainers"):
        value = pod_spec.get(key)
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    containers.append((f"{key}[{index}]", item))
    return containers


def _metadata_name(document: Any) -> str | None:
    value = _value_at(document, "metadata", "name")
    return value if isinstance(value, str) and value else None


def _required_group_label(group: tuple[tuple[str, ...], ...]) -> str:
    if len(group) == 1:
        return ".".join(group[0])
    return " OR ".join(".".join(path) for path in group)


def _level_2_errors(documents: tuple[Any, ...]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if not documents:
        return [_error(2, "kubernetes_identity", "manifest has no YAML documents")]

    known_kinds = set(KNOWN_KINDS)
    for index, document in enumerate(documents):
        prefix = f"documents[{index}]"
        if not isinstance(document, dict):
            errors.append(_error(2, "kubernetes_identity", "document is not a mapping", path=prefix))
            continue

        api_version = document.get("apiVersion")
        if not isinstance(api_version, str) or not api_version:
            errors.append(_error(2, "kubernetes_identity", "missing non-empty apiVersion", path=f"{prefix}.apiVersion"))

        kind = document.get("kind")
        if not isinstance(kind, str) or not kind:
            errors.append(_error(2, "kubernetes_identity", "missing non-empty kind", path=f"{prefix}.kind"))
            continue
        if kind not in known_kinds:
            errors.append(_error(2, "kubernetes_identity", f"unknown kind {kind!r}", path=f"{prefix}.kind"))

        if _metadata_name(document) is None:
            errors.append(_error(2, "kubernetes_identity", "missing non-empty metadata.name", path=f"{prefix}.metadata.name"))

        for group in GENERIC_REQUIRED_FIELD_GROUPS + KIND_REQUIRED_FIELD_GROUPS.get(kind, ()):
            if not any(_path_present(document, path) for path in group):
                errors.append(
                    _error(
                        2,
                        "required_field",
                        f"missing required field group {_required_group_label(group)}",
                        path=prefix,
                    )
                )
    return errors


def _valid_cron_expression(value: str) -> bool:
    stripped = value.strip()
    if stripped.startswith("@"):
        return stripped in {"@yearly", "@annually", "@monthly", "@weekly", "@daily", "@hourly", "@reboot"}
    return len(stripped.split()) == 5


def _level_3_errors(documents: tuple[Any, ...]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        kind = document.get("kind")
        prefix = f"documents[{index}]"

        if kind in WORKLOAD_KINDS:
            selector = _mapping_at(document, "spec", "selector", "matchLabels")
            labels = _mapping_at(document, "spec", "template", "metadata", "labels")
            if selector is not None and labels is not None:
                for key, value in selector.items():
                    if labels.get(key) != value:
                        errors.append(
                            _error(
                                3,
                                "selector_template_mismatch",
                                f"selector label {key!r}={value!r} is not present in template labels",
                                path=f"{prefix}.spec.selector.matchLabels",
                            )
                        )

        if kind == "CronJob":
            schedule = _value_at(document, "spec", "schedule")
            if isinstance(schedule, str) and not _valid_cron_expression(schedule):
                errors.append(_error(3, "invalid_schedule", "CronJob schedule is not a five-field or @ cron expression", path=f"{prefix}.spec.schedule"))

        min_replicas = _value_at(document, "spec", "minReplicas")
        max_replicas = _value_at(document, "spec", "maxReplicas")
        if isinstance(min_replicas, int) and isinstance(max_replicas, int) and min_replicas > max_replicas:
            errors.append(_error(3, "invalid_replica_range", "minReplicas is greater than maxReplicas", path=f"{prefix}.spec"))

        service_ports = _value_at(document, "spec", "ports")
        if isinstance(service_ports, list):
            for port_index, port_row in enumerate(service_ports):
                port = port_row.get("port") if isinstance(port_row, dict) else None
                if not isinstance(port, int) or not 1 <= port <= 65535:
                    errors.append(_error(3, "invalid_port", "Service port must be an integer between 1 and 65535", path=f"{prefix}.spec.ports[{port_index}].port"))

        for pod_path, pod_spec in _pod_specs(document):
            volumes = pod_spec.get("volumes")
            volume_rows = volumes if isinstance(volumes, list) else []
            volume_names = {
                volume.get("name")
                for volume in volume_rows
                if isinstance(volume, dict) and isinstance(volume.get("name"), str)
            }
            for container_path, container in _container_lists(pod_spec):
                if not isinstance(container.get("image"), str) or not container.get("image"):
                    errors.append(_error(3, "container_missing_image", "container is missing a non-empty image", path=f"{prefix}.{pod_path}.{container_path}.image"))

                ports = container.get("ports")
                if isinstance(ports, list):
                    for port_index, port_row in enumerate(ports):
                        container_port = port_row.get("containerPort") if isinstance(port_row, dict) else None
                        if not isinstance(container_port, int) or not 1 <= container_port <= 65535:
                            errors.append(_error(3, "invalid_port", "containerPort must be an integer between 1 and 65535", path=f"{prefix}.{pod_path}.{container_path}.ports[{port_index}].containerPort"))

                mounts = container.get("volumeMounts")
                if isinstance(mounts, list):
                    for mount_index, mount in enumerate(mounts):
                        mount_name = mount.get("name") if isinstance(mount, dict) else None
                        if isinstance(mount_name, str) and mount_name not in volume_names:
                            errors.append(
                                _error(
                                    3,
                                    "volume_mount_without_volume",
                                    f"volumeMount {mount_name!r} has no matching volume",
                                    path=f"{prefix}.{pod_path}.{container_path}.volumeMounts[{mount_index}].name",
                                )
                            )
    return errors


def _workload_labels(documents: tuple[Any, ...]) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for document in documents:
        if not isinstance(document, dict):
            continue
        mapping = _mapping_at(document, "spec", "template", "metadata", "labels")
        if mapping is not None:
            labels.append(mapping)
        if document.get("kind") == "Pod":
            pod_labels = _mapping_at(document, "metadata", "labels")
            if pod_labels is not None:
                labels.append(pod_labels)
    return labels


def _defined_names_by_kind(documents: tuple[Any, ...]) -> dict[str, set[str]]:
    names: dict[str, set[str]] = {}
    for document in documents:
        if not isinstance(document, dict):
            continue
        kind = document.get("kind")
        name = _metadata_name(document)
        if isinstance(kind, str) and name:
            names.setdefault(kind, set()).add(name)
    return names


def _reference_exists_when_resolvable(
    errors: list[dict[str, Any]],
    *,
    level: int,
    category: str,
    names_by_kind: dict[str, set[str]],
    kind: str,
    name: Any,
    path: str,
) -> None:
    if not isinstance(name, str) or not name:
        return
    known_names = names_by_kind.get(kind)
    if known_names and name not in known_names:
        errors.append(_error(level, category, f"{kind} reference {name!r} does not match a local {kind}", path=path))


def _level_4_errors(documents: tuple[Any, ...]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    names_by_kind = _defined_names_by_kind(documents)
    workload_labels = _workload_labels(documents)

    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        prefix = f"documents[{index}]"
        if document.get("kind") == "Service":
            selector = _mapping_at(document, "spec", "selector")
            if selector is not None and not any(
                all(labels.get(key) == value for key, value in selector.items())
                for labels in workload_labels
            ):
                errors.append(_error(4, "service_selector_without_workload", "Service selector does not match any local workload labels", path=f"{prefix}.spec.selector"))

        if document.get("kind") == "RoleBinding":
            role_ref = _mapping_at(document, "roleRef")
            subjects = document.get("subjects")
            if role_ref is None:
                errors.append(_error(4, "rbac_incomplete", "RoleBinding is missing roleRef", path=f"{prefix}.roleRef"))
            else:
                for field in ("apiGroup", "kind", "name"):
                    if not isinstance(role_ref.get(field), str) or not role_ref.get(field):
                        errors.append(_error(4, "rbac_incomplete", f"RoleBinding roleRef is missing {field}", path=f"{prefix}.roleRef.{field}"))
                if role_ref.get("kind") == "Role":
                    _reference_exists_when_resolvable(
                        errors,
                        level=4,
                        category="rbac_reference_missing",
                        names_by_kind=names_by_kind,
                        kind="Role",
                        name=role_ref.get("name"),
                        path=f"{prefix}.roleRef.name",
                    )
            if not isinstance(subjects, list) or not subjects:
                errors.append(_error(4, "rbac_incomplete", "RoleBinding is missing non-empty subjects", path=f"{prefix}.subjects"))

        for pod_path, pod_spec in _pod_specs(document):
            _reference_exists_when_resolvable(
                errors,
                level=4,
                category="service_account_reference_missing",
                names_by_kind=names_by_kind,
                kind="ServiceAccount",
                name=pod_spec.get("serviceAccountName"),
                path=f"{prefix}.{pod_path}.serviceAccountName",
            )

            for volume_index, volume in enumerate(pod_spec.get("volumes", []) if isinstance(pod_spec.get("volumes"), list) else []):
                if not isinstance(volume, dict):
                    continue
                _reference_exists_when_resolvable(
                    errors,
                    level=4,
                    category="configmap_reference_missing",
                    names_by_kind=names_by_kind,
                    kind="ConfigMap",
                    name=_value_at(volume, "configMap", "name"),
                    path=f"{prefix}.{pod_path}.volumes[{volume_index}].configMap.name",
                )
                _reference_exists_when_resolvable(
                    errors,
                    level=4,
                    category="secret_reference_missing",
                    names_by_kind=names_by_kind,
                    kind="Secret",
                    name=_value_at(volume, "secret", "secretName"),
                    path=f"{prefix}.{pod_path}.volumes[{volume_index}].secret.secretName",
                )
                _reference_exists_when_resolvable(
                    errors,
                    level=4,
                    category="pvc_reference_missing",
                    names_by_kind=names_by_kind,
                    kind="PersistentVolumeClaim",
                    name=_value_at(volume, "persistentVolumeClaim", "claimName"),
                    path=f"{prefix}.{pod_path}.volumes[{volume_index}].persistentVolumeClaim.claimName",
                )
    return errors


def _image_uses_latest(image: str) -> bool:
    last_segment = image.rsplit("/", maxsplit=1)[-1]
    return ":" not in last_segment or last_segment.endswith(":latest")


def _resource_value(container: dict[str, Any], section: str, resource: str) -> Any:
    return _value_at(container, "resources", section, resource)


def _level_5_errors(documents: tuple[Any, ...]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            continue
        prefix = f"documents[{index}]"
        for key in ("hostNetwork", "hostPID", "hostIPC"):
            if document.get("spec", {}).get(key) is True if isinstance(document.get("spec"), dict) else False:
                errors.append(_error(5, "unsafe_host_namespace", f"{key} is enabled", path=f"{prefix}.spec.{key}"))

        for pod_path, pod_spec in _pod_specs(document):
            pod_security_context = pod_spec.get("securityContext") if isinstance(pod_spec.get("securityContext"), dict) else {}
            for container_path, container in _container_lists(pod_spec):
                for section in ("requests", "limits"):
                    for resource in ("cpu", "memory"):
                        if _resource_value(container, section, resource) is None:
                            errors.append(_error(5, "missing_resource_requirement", f"container is missing resources.{section}.{resource}", path=f"{prefix}.{pod_path}.{container_path}.resources.{section}.{resource}"))

                image = container.get("image")
                if isinstance(image, str) and _image_uses_latest(image):
                    errors.append(_error(5, "latest_image_tag", "container image uses an implicit or explicit latest tag", path=f"{prefix}.{pod_path}.{container_path}.image"))

                container_security_context = container.get("securityContext") if isinstance(container.get("securityContext"), dict) else {}
                run_as_non_root = container_security_context.get("runAsNonRoot", pod_security_context.get("runAsNonRoot"))
                if run_as_non_root is not True:
                    errors.append(_error(5, "missing_run_as_non_root", "container does not explicitly set runAsNonRoot: true", path=f"{prefix}.{pod_path}.{container_path}.securityContext.runAsNonRoot"))
                if container_security_context.get("allowPrivilegeEscalation") is True:
                    errors.append(_error(5, "privilege_escalation", "container enables allowPrivilegeEscalation", path=f"{prefix}.{pod_path}.{container_path}.securityContext.allowPrivilegeEscalation"))
                if container_security_context.get("privileged") is True:
                    errors.append(_error(5, "privileged_container", "container is privileged", path=f"{prefix}.{pod_path}.{container_path}.securityContext.privileged"))
                if container_security_context.get("readOnlyRootFilesystem") is not True:
                    errors.append(_error(5, "missing_read_only_root_filesystem", "container does not explicitly set readOnlyRootFilesystem: true", path=f"{prefix}.{pod_path}.{container_path}.securityContext.readOnlyRootFilesystem"))
    return errors


def evaluate_kubernetes_domain_validity(
    documents: tuple[Any, ...],
    *,
    yaml_parse_ok: bool,
    block_parse_ok: bool,
    valid_block_ratio: float,
    document_index_monotonic_ok: bool,
    line_index_sequence_ok: bool,
    indentation_leak_rate: float,
) -> KubernetesDomainEvaluation:
    errors: list[dict[str, Any]] = []
    level_scores: dict[str, float | None] = {
        "level_0_yaml_parse": 1.0 if yaml_parse_ok else 0.0,
        "level_1_block_contract": None,
        "level_2_kubernetes_identity": None,
        "level_3_intra_resource_invariants": None,
        "level_4_inter_resource_invariants": None,
        "level_5_static_quality_smells": None,
    }

    if not yaml_parse_ok:
        errors.append(_error(0, "yaml_parse", "prediction YAML is not parseable"))
    level_1_ok = (
        yaml_parse_ok
        and block_parse_ok
        and valid_block_ratio == 1.0
        and document_index_monotonic_ok
        and line_index_sequence_ok
        and indentation_leak_rate == 0.0
    )
    level_scores["level_1_block_contract"] = 1.0 if level_1_ok else 0.0
    if yaml_parse_ok and not level_1_ok:
        errors.append(_error(1, "block_contract", "parser-facing block contract is not fully satisfied"))

    if yaml_parse_ok:
        for level, key, checker in (
            (2, "level_2_kubernetes_identity", _level_2_errors),
            (3, "level_3_intra_resource_invariants", _level_3_errors),
            (4, "level_4_inter_resource_invariants", _level_4_errors),
            (5, "level_5_static_quality_smells", _level_5_errors),
        ):
            level_errors = checker(documents)
            errors.extend(level_errors)
            level_scores[key] = 0.0 if level_errors else 1.0
    else:
        for key in list(level_scores)[2:]:
            level_scores[key] = 0.0

    validity_level = -1
    for index, value in enumerate(level_scores.values()):
        if value is None or value >= 1.0:
            validity_level = index
            continue
        break

    numeric_scores = [value for value in level_scores.values() if value is not None]
    validity_score = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.0
    return KubernetesDomainEvaluation(
        kubernetes_domain_validity_level=validity_level,
        kubernetes_domain_gate_pass=validity_level == 5,
        kubernetes_domain_validity_score=validity_score,
        kubernetes_domain_level_scores=level_scores,
        kubernetes_domain_errors=tuple(errors),
    )

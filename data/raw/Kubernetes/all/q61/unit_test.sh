kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=my-ds --timeout=60s
cpu_limit=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')
memory_limit=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')
volume_mount=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].volumeMounts[0].mountPath}')

[ "$cpu_limit" = "500m" ] && \
[ "$memory_limit" = "100Mi" ] && \
[ "$volume_mount" = "/var/shared-data" ] && \
echo cloudeval_unit_test_passed

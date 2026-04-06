kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=my-ds --timeout=60s

if ! kubectl get configmap nginx-config; then
  echo "ConfigMap nginx-config does not exist."
  exit 1
fi

config_volume_mount=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].volumeMounts[?(@.name=="nginx-config-volume")].mountPath}')
[ "$config_volume_mount" = "/etc/nginx/config" ] && \
echo cloudeval_unit_test_passed

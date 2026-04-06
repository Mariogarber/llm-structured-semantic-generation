kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l name=my-ds --timeout=60s
node_selector=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.nodeSelector.kubernetes\.io/os}')
liveness_path=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].livenessProbe.httpGet.path}')
readiness_path=$(kubectl get ds my-ds -o=jsonpath='{.spec.template.spec.containers[0].readinessProbe.httpGet.path}')

[ "$node_selector" = "linux" ] && \
[ "$liveness_path" = "/" ] && \
[ "$readiness_path" = "/" ] && \
echo cloudeval_unit_test_passed

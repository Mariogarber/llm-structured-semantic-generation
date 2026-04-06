kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/hello-limited --timeout=60s
kubectl get pod hello-limited -o=jsonpath='{.spec.containers[0].resources.requests.cpu}' | grep -w "250m" && \
kubectl get pod hello-limited -o=jsonpath='{.spec.containers[0].resources.requests.memory}' | grep -w "128Mi" && \
kubectl get pod hello-limited -o=jsonpath='{.spec.containers[0].resources.limits.cpu}' | grep -w "500m" && \
kubectl get pod hello-limited -o=jsonpath='{.spec.containers[0].resources.limits.memory}' | grep -w "256Mi" && \
echo cloudeval_unit_test_passed

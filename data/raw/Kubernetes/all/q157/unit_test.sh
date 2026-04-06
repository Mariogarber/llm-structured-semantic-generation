kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/http-server --timeout=60s

kubectl get pods | grep "http-server" | grep "Running" && \
kubectl describe pod http-server | grep "Image:" | grep "kuard-amd64" && \
kubectl describe pod http-server | grep "Port:" | grep "8080/TCP" && \
echo cloudeval_unit_test_passed

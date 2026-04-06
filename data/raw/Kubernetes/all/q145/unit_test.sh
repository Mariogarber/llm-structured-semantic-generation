kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/nginx-server --timeout=30s

kubectl get pods | grep "nginx-server" | grep "Running" && \
kubectl describe pod nginx-server | grep "Image:" | grep "nginx" && \
kubectl describe pod nginx-server | grep "Port:" | grep "80/TCP" && \
echo cloudeval_unit_test_passed

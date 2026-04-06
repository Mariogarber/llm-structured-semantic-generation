kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=kuard-pod --timeout=60s
[ $(kubectl get pods -l app=kuard-pod | grep -o "Running" | wc -l) -eq 1 ] && \
echo cloudeval_unit_test_passed

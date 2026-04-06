kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pods -l app=test-pod --timeout=60s

kubectl get rc test-rc | grep -q 1 && \
echo cloudeval_unit_test_passed

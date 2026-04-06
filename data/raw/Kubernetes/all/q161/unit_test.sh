kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/liveness-http --timeout=60s
sleep 15
kubectl describe pod liveness-http | grep "Liveness probe failed" || exit 1
sleep 5
kubectl describe pod liveness-http | grep "Killing" && \
echo cloudeval_unit_test_passed

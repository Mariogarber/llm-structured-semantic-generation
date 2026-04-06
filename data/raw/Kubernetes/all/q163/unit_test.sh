kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod emptydir-pod --timeout=60s

[ "$(kubectl exec emptydir-pod -- cat /tmp/volume/message.txt)" = "Hello World" ] && \
echo cloudeval_unit_test_passed

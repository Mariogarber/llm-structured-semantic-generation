kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod/secret-pod --timeout=60s

kubectl logs pod/secret-pod | grep -q "secret-file" && \
kubectl exec secret-pod -- ls -l /etc/secret-volume | grep -q "secret-file" && \
kubectl exec secret-pod -- cat /etc/secret-volume/secret-file | grep -q "value-1" && \
echo cloudeval_unit_test_passed

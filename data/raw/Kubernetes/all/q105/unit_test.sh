kubectl apply -f labeled_code.yaml
sleep 15

kubectl get job failjob -o=jsonpath='{.status.conditions[*].message}' | grep -q "failed with exit code 42" && \
kubectl get job failjob -o=jsonpath='{.status.conditions[*].reason}' | grep -q "PodFailurePolicy" && \
echo cloudeval_unit_test_passed

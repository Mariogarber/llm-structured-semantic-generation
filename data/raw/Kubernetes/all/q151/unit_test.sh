kubectl apply -f labeled_code.yaml
sleep 5

kubectl get pod scheduled-pod -o jsonpath='{.spec.schedulingGates}' | grep -q "example.com/foo" && \
kubectl get pod scheduled-pod -o jsonpath='{.spec.schedulingGates}' | grep -q "example.com/bar" && \
echo cloudeval_unit_test_passed

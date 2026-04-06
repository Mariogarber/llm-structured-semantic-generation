kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/sleep-pod --timeout=60s
CMD=$(kubectl get pod sleep-pod -o=jsonpath='{.spec.containers[0].command}')

echo $CMD | grep -q 'sleep $(SLEEP_TIME)' && \
echo cloudeval_unit_test_passed

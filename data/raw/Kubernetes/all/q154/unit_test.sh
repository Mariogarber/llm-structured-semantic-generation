kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/cap-pod --timeout=60s
caps=$(kubectl get pod cap-pod -o=jsonpath='{.spec.containers[0].securityContext.capabilities.add}')

echo $caps | grep -q "AUDIT_WRITE" && \
echo $caps | grep -q "CHOWN" && \
echo cloudeval_unit_test_passed

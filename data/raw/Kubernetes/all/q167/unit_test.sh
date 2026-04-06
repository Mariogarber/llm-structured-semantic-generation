kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/default-pod --timeout=60s

[ "$(kubectl get pod default-pod -o jsonpath='{.spec.securityContext.seccompProfile.type}')" = "RuntimeDefault" ] && \
[ "$(kubectl get pod default-pod -o jsonpath='{.spec.containers[0].securityContext.allowPrivilegeEscalation}')" = "false" ] && \
kubectl logs pod/default-pod | grep -q "Operation not permitted" && \
echo cloudeval_unit_test_passed

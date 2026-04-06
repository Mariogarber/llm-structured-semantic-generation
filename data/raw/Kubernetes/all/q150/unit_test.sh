kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/pod-nosec --timeout=60s

[ "$(kubectl get pod pod-nosec -o=jsonpath='{.spec.volumes[0].secret.secretName}')" = "optional-secret" ] && \
[ "$(kubectl get pod pod-nosec -o=jsonpath='{.spec.volumes[0].secret.optional}')" = "true" ] && \
! kubectl describe pods/pod-nosec | grep -q "not found" && \
echo cloudeval_unit_test_passed

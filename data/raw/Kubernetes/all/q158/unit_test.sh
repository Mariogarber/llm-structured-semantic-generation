kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/sysctls1 --timeout=60s
sysctl_value=$(kubectl get pod sysctls1 -o=jsonpath='{.spec.securityContext.sysctls[0].value}')

[ "$sysctl_value" = "0" ] && \
echo cloudeval_unit_test_passed

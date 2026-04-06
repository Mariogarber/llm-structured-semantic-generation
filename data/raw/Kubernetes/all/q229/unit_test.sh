kubectl apply -f labeled_code.yaml

[ "$(kubectl get secret tls-secret -o=jsonpath='{.type}')" = "kubernetes.io/tls" ] && \
echo cloudeval_unit_test_passed

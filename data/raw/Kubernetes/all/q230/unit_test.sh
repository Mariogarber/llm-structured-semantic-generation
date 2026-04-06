kubectl apply -f labeled_code.yaml

[ "$(kubectl get secret bs-sec -o=jsonpath='{.type}')" = "bootstrap.kubernetes.io/token" ] && \
[ "$(kubectl get secret bs-sec -o=jsonpath='{.data.token-id}' | base64 --decode)" = "abcdef" ] && \
[ "$(kubectl get secret bs-sec -o=jsonpath='{.data.token-secret}' | base64 --decode)" = "0123456789abcdef" ] && \
[ "$(kubectl get secret bs-sec -o=jsonpath='{.data.expiration}' | base64 --decode)" = "2023-09-21T23:59:00Z" ] && \
[ "$(kubectl get secret bs-sec -o=jsonpath='{.data.usage-bootstrap-authentication}' | base64 --decode)" = "true" ] && \
[ "$(kubectl get secret bs-sec -o=jsonpath='{.data.usage-bootstrap-signing}' | base64 --decode)" = "true" ] && \
echo cloudeval_unit_test_passed

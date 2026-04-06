kubectl apply -f labeled_code.yaml

[ "$(kubectl get secret mysec -o=jsonpath='{.data.username}' | base64 -d)" = "admin" ] && \
[ "$(kubectl get secret mysec -o=jsonpath='{.data.password}' | base64 -d)" = "sec123" ] && \
echo cloudeval_unit_test_passed

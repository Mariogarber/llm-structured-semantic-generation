kubectl apply -f labeled_code.yaml

kubectl get ing | grep -q "ana-ing" && \
[ "$(kubectl get ing ana-ing -o jsonpath='{.spec.rules[0].http.paths[0].backend.service.name}')" = "ana-dash" ] &&
[ "$(kubectl get ing ana-ing -o jsonpath='{.spec.rules[0].http.paths[0].backend.service.port.number}')" = "8085" ] &&
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml
sleep 3

name=$(kubectl get ing my-ing -o=jsonpath='{.spec.rules[0].http.paths[0].backend.service.name}')
port=$(kubectl get ing my-ing -o=jsonpath='{.spec.rules[0].http.paths[0].backend.service.port.number}')

[ "$name" = "nginx-svc" ] && \
[ "$port" = "443" ] && \
echo cloudeval_unit_test_passed

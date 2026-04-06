kubectl apply -f labeled_code.yaml

name_1=$(kubectl get ing my-ing2 -o=jsonpath='{.spec.rules[0].http.paths[0].backend.service.name}')
port_1=$(kubectl get ing my-ing2 -o=jsonpath='{.spec.rules[0].http.paths[0].backend.service.port.number}')
name_2=$(kubectl get ing my-ing2 -o=jsonpath='{.spec.rules[0].http.paths[1].backend.service.name}')
port_2=$(kubectl get ing my-ing2 -o=jsonpath='{.spec.rules[0].http.paths[1].backend.service.port.number}')
tls_secret=$(kubectl get ing my-ing2 -o=jsonpath='{.spec.tls[0].secretName}')

[ "$name_1" = "svc-1" ] && \
[ "$port_1" = "8080" ] && \
[ "$name_2" = "svc-2" ] && \
[ "$port_2" = "9090" ] && \
[ "$tls_secret" = "example-tls" ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml

[ "$(kubectl get svc wordpress -o jsonpath='{.spec.type}')" = "LoadBalancer" ] && \
[ "$(kubectl get svc wordpress -o jsonpath='{.spec.ports[0].port}')" = "80" ] && \
echo cloudeval_unit_test_passed

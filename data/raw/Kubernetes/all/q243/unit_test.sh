kubectl apply -f labeled_code.yaml

[ "$(kubectl get svc frontend-service -o=jsonpath='{.spec.selector.app}')" = "frontend" ] && \
[ "$(kubectl get svc frontend-service -o=jsonpath='{.spec.ports[0].port}')" = "80" ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml

[ "$(kubectl get svc wordpress-mysql -o jsonpath='{.spec.ports[0].port}')" -eq 3306 ] && \
echo cloudeval_unit_test_passed

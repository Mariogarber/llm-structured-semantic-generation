kubectl apply -f labeled_code.yaml
sleep 1

[ "$(kubectl get hpa simple-web-app-hpa -o=jsonpath='{.spec.metrics[?(@.resource.name=="cpu")].resource.target.averageUtilization}')" = "60" ] &&
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml

kubectl get ing | grep -q "fruit-ing" && \
[ $(kubectl get ing fruit-ing -o=jsonpath='{.metadata.annotations.ingress\.kubernetes\.io/rewrite-target}') = "/" ] && \
kubectl get ing fruit-ing -o=jsonpath='{.spec.rules[0].http.paths[?(@.backend.service.name=="apple-svc")].backend.service.port.number}' | grep -q '4567' && \
kubectl get ing fruit-ing -o=jsonpath='{.spec.rules[0].http.paths[?(@.backend.service.name=="banana-svc")].backend.service.port.number}' | grep -q '6789' && \
echo cloudeval_unit_test_passed

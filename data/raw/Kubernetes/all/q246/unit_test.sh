kubectl apply -f labeled_code.yaml

kubectl get svc | grep -q "my-nginx-svc" && \
[ "$(kubectl get service my-nginx-svc -o=jsonpath='{.spec.type}')" = "ClusterIP" ] && \
[ "$(kubectl get service my-nginx-svc -o=jsonpath='{.spec.ports[0].port}')" = "80" ] && \
[ "$(kubectl get service my-nginx-svc -o=jsonpath='{.spec.selector.app}')" = "my-nginx" ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml
sleep 3

kubectl get svc -o jsonpath='{.items[*].metadata.name}' | grep -q "webapp-lb" && \
[ "$(kubectl get svc webapp-lb -o jsonpath='{.spec.type}')" = "LoadBalancer" ] && \
[ "$(kubectl get svc webapp-lb -o jsonpath='{.spec.ports[*].port}')" -eq 80 ] && \
[ "$(kubectl get svc webapp-lb -o jsonpath='{.spec.ports[*].targetPort}')" -eq 8080 ] && \
[ "$(kubectl get svc webapp-lb -o jsonpath='{.spec.selector.app}')" = "webapp" ] && \
echo cloudeval_unit_test_passed

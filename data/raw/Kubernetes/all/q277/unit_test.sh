kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=my-cassandra --timeout=60s

[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')" = "500m" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')" = "1Gi" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')" = "500m" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')" = "1Gi" ] && \
echo cloudeval_unit_test_passed

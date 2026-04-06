kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=my-cassandra --timeout=60s

[ "$(kubectl get sts my-cassandra -o jsonpath='{.metadata.name}')" = "my-cassandra" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.replicas}')" = "3" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="MAX_HEAP_SIZE")].value}')" = "512M" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].env[?(@.name=="HEAP_NEWSIZE")].value}')" = "100M" ] && \
echo cloudeval_unit_test_passed

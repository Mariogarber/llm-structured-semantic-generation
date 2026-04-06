kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=my-cassandra --timeout=60s

[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].ports[0].containerPort}')" = "7000" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].ports[0].name}')" = "intra-node" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].ports[1].containerPort}')" = "9042" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].ports[1].name}')" = "cql" ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=my-cassandra --timeout=60s

[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.terminationGracePeriodSeconds}')" = "1800" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].imagePullPolicy}')" = "Always" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.template.spec.containers[0].securityContext.capabilities.add[0]}')" = "IPC_LOCK" ] && \
echo cloudeval_unit_test_passed

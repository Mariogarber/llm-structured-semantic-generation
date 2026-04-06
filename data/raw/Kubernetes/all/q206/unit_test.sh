kubectl apply -f labeled_code.yaml
sleep 5
kubectl wait --for=condition=ready pod -l app=redis --timeout=60s
rc_replicas=$(kubectl get rc redis-replica -o=jsonpath='{.spec.replicas}')
env_vars=$(kubectl get rc redis-replica -o=jsonpath='{.spec.template.spec.containers[0].env}')
rc_detail=$(kubectl get rc redis-replica -o json)

[ "$rc_replicas" -eq 2 ] && \
echo "$env_vars" | grep -q "GET_HOSTS_FROM" && \
echo "$env_vars" | grep -q "env" && \
[ "$(kubectl get rc redis-replica -o=jsonpath='{.spec.template.spec.containers[0].ports[0].containerPort}')" -eq 6380 ] && \
[ "$(kubectl get rc redis-replica -o=jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')" = "200Mi" ] && \
[ "$(kubectl get rc redis-replica -o=jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')" = "200m" ] && \
echo cloudeval_unit_test_passed

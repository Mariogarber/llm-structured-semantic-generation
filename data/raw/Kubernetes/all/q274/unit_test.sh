kubectl apply -f labeled_code.yaml
sleep 30
kubectl wait --for=condition=ready pod -l app=nginx --timeout=60s

replica_count=$(kubectl get sts web -o=jsonpath='{.spec.replicas}')
min_ready_seconds=$(kubectl get sts web -o=jsonpath='{.spec.minReadySeconds}')
pod_name=$(kubectl get pods -l app=nginx -o jsonpath='{.items[0].metadata.name}')
pod_image=$(kubectl get pod $pod_name -o=jsonpath='{.spec.containers[0].image}')

[ "$replica_count" -eq 2 ] && \
[ "$min_ready_seconds" -eq 10 ] && \
[ "$pod_image" = "k8s.gcr.io/nginx-slim:0.16" ] && \
echo cloudeval_unit_test_passed

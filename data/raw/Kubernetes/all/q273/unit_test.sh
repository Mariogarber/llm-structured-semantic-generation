kubectl apply -f labeled_code.yaml
sleep 15
kubectl wait --for=condition=ready pod -l app=nginx --timeout=60s

replica_count=$(kubectl get sts web -o=jsonpath='{.spec.replicas}')
pvc_sc=$(kubectl get pvc -l app=nginx -o=jsonpath='{.items[0].spec.storageClassName}')
pod_name=$(kubectl get pods -l app=nginx -o jsonpath='{.items[0].metadata.name}')
pod_image=$(kubectl get pod $pod_name -o=jsonpath='{.spec.containers[0].image}')

[ "$replica_count" -eq 3 ] && \
[ "$pvc_sc" = "standard" ] && \
[ "$pod_image" = "k8s.gcr.io/nginx-slim:0.8" ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml
kubectl -n kube-system wait --for=condition=ready pod -l name=fluentd-worker --timeout=60s

daemonset_name=$(kubectl -n kube-system get ds -o=jsonpath='{.items[?(@.metadata.labels.k8s-app=="fluentd-logging")].metadata.name}')
pod_name=$(kubectl -n kube-system get pods -l name=fluentd-worker -o jsonpath='{.items[0].metadata.name}')
pod_image=$(kubectl -n kube-system get pod $pod_name -o=jsonpath='{.spec.containers[0].image}')

[ "$daemonset_name" = "fluentd-worker" ] && \
[ "$pod_image" = "quay.io/fluentd_elasticsearch/fluentd:v2.5.2" ] && \
echo cloudeval_unit_test_passed

kubectl delete -n kube-system ds fluentd-worker

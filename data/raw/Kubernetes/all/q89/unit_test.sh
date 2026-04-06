kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l run=php-apache --timeout=60s
sleep 30
min_replicas=$(kubectl get hpa php-apache -o=jsonpath='{.spec.minReplicas}')
max_replicas=$(kubectl get hpa php-apache -o=jsonpath='{.spec.maxReplicas}')
cur_replicas=$(kubectl get hpa php-apache -o=jsonpath='{.status.currentReplicas}')

[ "$min_replicas" = "2" ] && \
[ "$max_replicas" = "10" ] && \
[ "$cur_replicas" -ge "$min_replicas" ] && \
[ "$cur_replicas" -le "$max_replicas" ] && \
echo cloudeval_unit_test_passed

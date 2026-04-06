kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod/local-pvc-pod-1 --timeout=60s
kubectl delete pod local-pvc-pod-1
kubectl wait --for=condition=ready pod/local-pvc-pod-2 --timeout=60s

[ "$(kubectl exec local-pvc-pod-2 -- cat /mnt/volume/message.txt)" = "It is persistent!" ] && \
echo cloudeval_unit_test_passed

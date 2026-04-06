kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod local-pvc-pod --timeout=60s

kubectl exec local-pvc-pod -- cat /mnt/volume/message.txt | grep -q "persistent" && \
echo cloudeval_unit_test_passed

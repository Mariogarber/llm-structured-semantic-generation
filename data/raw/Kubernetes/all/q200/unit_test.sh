kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=nginx-rs-pod,env=dev --timeout=120s
[ $(kubectl get pods -l app=nginx-rs-pod,env=dev | grep -o "Running" | wc -l) -eq 2 ] && \
echo cloudeval_unit_test_passed

kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/nginx-token --timeout=15s
# kubectl describe pod nginx-token | grep "" && echo "Correct behavior, keep going" || { echo "Wrong behavior, stopping execution"; exit 1; }
# sleep 20
kubectl describe pod nginx-token | grep "TokenExpirationSeconds:  601" && echo cloudeval_unit_test_passed
# may not specify a duration less than 10 minutes, not practical to wait until it actually expire
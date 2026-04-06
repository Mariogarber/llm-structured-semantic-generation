kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/goproxy --timeout=15s
kubectl get pod goproxy -o json | grep "\"livenessProbe\": {" && kubectl get pod goproxy -o json | grep "\"readinessProbe\": {" &&  echo cloudeval_unit_test_passed

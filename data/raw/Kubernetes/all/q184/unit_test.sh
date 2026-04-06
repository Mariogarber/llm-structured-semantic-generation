kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/two-containers --timeout=30s
# kubectl get pod two-containers
kubectl exec -it two-containers -c nginx-container -- /bin/bash -c "curl localhost && exit" | grep "Hello from the debian container" && echo cloudeval_unit_test_passed
# INCLUDE: "Hello from the debian container"

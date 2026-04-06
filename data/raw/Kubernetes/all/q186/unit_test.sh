kubectl delete configmap special-config
kubectl create configmap special-config --from-literal=SPECIAL_LEVEL=LevelValue
kubectl wait --for=condition=complete configmap/special-config  --timeout=5s
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/config-pod-1  --timeout=30s
kubectl get pods config-pod-1
kubectl logs config-pod-1 | grep "SPECIAL_LEVEL_KEY=LevelValue" && echo cloudeval_unit_test_passed
# INCLUDE: "SPECIAL_LEVEL_KEY=LevelValue"


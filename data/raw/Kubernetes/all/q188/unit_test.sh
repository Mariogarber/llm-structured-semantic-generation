kubectl delete configmap special-config
kubectl create configmap special-config --from-literal=SPECIAL_LEVEL_1=LevelValue1 --from-literal=SPECIAL_LEVEL_2=LevelValue2
kubectl wait --for=condition=complete configmap/special-config  --timeout=5s
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=running pods/config-pod-4  --timeout=30s
kubectl exec -it config-pod-4 -- /bin/sh -c "ls /etc/config && exit" | grep "SPECIAL_LEVEL_1" && kubectl exec -it config-pod-4 -- /bin/sh -c "ls /etc/config && exit" | grep "SPECIAL_LEVEL_2" && echo cloudeval_unit_test_passed
# INCLUDE: "SPECIAL_LEVEL_1" "SPECIAL_LEVEL_2"

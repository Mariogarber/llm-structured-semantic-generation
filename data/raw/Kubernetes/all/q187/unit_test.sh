kubectl delete configmap special-configs
kubectl create configmap special-configs --from-literal=SPECIAL_LEVEL_1=LevelValue1 --from-literal=SPECIAL_LEVEL_2=LevelValue2 --from-literal=SPECIAL_LEVEL_3=LevelValue3 --from-literal=SPECIAL_LEVEL_4=LevelValue4
kubectl wait --for=condition=complete configmap/special-configs   --timeout=5s
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete pods/config-pod-2  --timeout=30s
kubectl get pods config-pod-2
kubectl logs config-pod-2 | grep "SPECIAL_LEVEL_1=LevelValue1" && kubectl logs config-pod-2 | grep "SPECIAL_LEVEL_2=LevelValue2" && kubectl logs config-pod-2 | grep "SPECIAL_LEVEL_3=LevelValue3" && kubectl logs config-pod-2 | grep "SPECIAL_LEVEL_4=LevelValue4" && echo cloudeval_unit_test_passed
# INCLUDE: "SPECIAL_LEVEL_1=LevelValue1" "SPECIAL_LEVEL_2=LevelValue2" "SPECIAL_LEVEL_3=LevelValue3" "SPECIAL_LEVEL_4=LevelValue4"

kubectl apply -f labeled_code.yaml
sleep 5
kubectl wait --for=condition=ready deployment --all --timeout=20s
pods=$(kubectl get pods --output=jsonpath={.items[0]..metadata.name})
kubectl exec $pods -- /bin/bash -c "ls -l && exit" | grep "tt" && echo cloudeval_unit_test_passed
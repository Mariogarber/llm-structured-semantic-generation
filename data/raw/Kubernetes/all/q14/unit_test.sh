kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=fluentd-elasticsearch --timeout=20s
pods=$(kubectl get pods --selector=name=fluentd-elasticsearch --output=jsonpath={.items..metadata.name})
resources=$(kubectl describe pod $pods)
echo $resources | grep "cpu: 100m" && echo $resources | grep "memory: 200Mi" && echo cloudeval_unit_test_passed
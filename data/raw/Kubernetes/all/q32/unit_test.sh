kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-configmap-volume --timeout=20s
pods=$(kubectl get pods --selector=name=ds-configmap-volume --output=jsonpath={.items..metadata.name})
kubectl exec $pods -- cat /config/config-data.txt | grep "This is data from the configmap." && echo cloudeval_unit_test_passed
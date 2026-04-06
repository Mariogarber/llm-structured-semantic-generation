kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-multi-container --timeout=20s
pods=$(kubectl get pods --selector=name=ds-multi-container --output=jsonpath={.items..metadata.name})
kubectl logs $pods -c container-1 | grep "Container 1 Running" && kubectl logs $pods -c container-2 | grep "Container 2 Running" && echo cloudeval_unit_test_passed

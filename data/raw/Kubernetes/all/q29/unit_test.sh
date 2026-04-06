kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-env-vars --timeout=20s
pods=$(kubectl get pods --selector=name=ds-env-vars --output=jsonpath={.items..metadata.name})
kubectl logs $pods | grep "Hello from the environment variable!" && echo cloudeval_unit_test_passed
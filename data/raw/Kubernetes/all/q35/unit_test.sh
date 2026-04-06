kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=ds-service-account-set --timeout=20s
pods=$(kubectl get pods --selector=name=ds-service-account-set --output=jsonpath={.items..metadata.name})
kubectl get serviceaccount ds-service-account && echo cloudeval_unit_test_passed1
kubectl logs $pods | grep "Running with service account" && echo cloudeval_unit_test_passed2
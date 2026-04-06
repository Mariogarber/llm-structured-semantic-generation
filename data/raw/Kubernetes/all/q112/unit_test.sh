kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/example --timeout=60s
pods=$(kubectl get pods --selector=job-name=example --output=jsonpath={.items..metadata.name})
kubectl logs $pods | grep "OK" && echo cloudeval_unit_test_passed_1
sleep 12
kubectl logs $pods 2>&1 | grep "not found" && echo cloudeval_unit_test_passed_2
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/example --timeout=60s
kubectl logs --selector job-name=example | grep "9216420199" && echo cloudeval_unit_test_passed
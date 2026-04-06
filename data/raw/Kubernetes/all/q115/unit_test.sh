kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/example --timeout=60s
kubectl logs --selector job-name=example | grep "0" && echo cloudeval_unit_test_passed_1
kubectl logs --selector job-name=example | grep "1" && echo cloudeval_unit_test_passed_2
kubectl logs --selector job-name=example | grep "2" && echo cloudeval_unit_test_passed_3
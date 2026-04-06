kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=DeadlineExceeded jobs/example --timeout=60s
kubectl logs --selector job-name=example | grep "icmp_seq" && echo cloudeval_unit_test_passed_1
kubectl events | grep "DeadlineExceeded" && echo cloudeval_unit_test_passed_2
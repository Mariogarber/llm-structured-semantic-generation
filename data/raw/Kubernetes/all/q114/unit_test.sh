kubectl apply -f labeled_code.yaml
sleep 30
kubectl get event | grep "Job has reached the specified backoff limit" && echo cloudeval_unit_test_passed
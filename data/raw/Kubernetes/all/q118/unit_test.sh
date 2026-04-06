kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/job-pod-failure-policy-failjob --timeout=40s
kubectl describe job job-pod-failure-policy-failjob | grep "Deleted pod: job-pod-failure-policy-failjob" && echo cloudeval_unit_test_passed
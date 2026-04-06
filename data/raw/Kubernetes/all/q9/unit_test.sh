kubectl apply -f labeled_code.yaml
sleep 70

kubectl describe cj x-job || grep "Concurrency" || grep "Replace" && \
kubectl get cj | grep "1" && \
echo cloudeval_unit_test_passed

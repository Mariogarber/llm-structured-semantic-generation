kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/hello-job-1
kubectl wait --for=condition=complete jobs/hello-job-2
pods=$(kubectl get pods --selector=job-name=hello-job-2 --output=jsonpath={.items..metadata.name})
kubectl logs $pods | grep "Hello" && echo cloudeval_unit_test_passed
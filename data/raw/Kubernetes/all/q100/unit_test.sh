kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=complete jobs/hello
pods=$(kubectl get pods --selector=job-name=hello --output=jsonpath={.items..metadata.name})
kubectl logs $pods | grep "Hello" && echo cloudeval_unit_test_passed
kubectl apply -f labeled_code.yaml
sleep 70
pods=$(kubectl get pods -o=jsonpath='{.items[0].metadata.name}')
kubectl logs $pods | grep "OK" && echo cloudeval_unit_test_passed
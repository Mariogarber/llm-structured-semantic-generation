kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod --all --timeout=20s
kubectl get svc my-service | egrep "my-service\s+LoadBalancer\s+[[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3}" && echo cloudeval_unit_test_passed
# from stackoverflow https://stackoverflow.com/questions/49981601/difference-between-targetport-and-port-in-kubernetes-service-definition
kubectl apply -f labeled_code.yaml
sleep 10
kubectl get svc my-service | egrep "my-service\s+ClusterIP\s+[[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3}\.[[:digit:]]{1,3}" && echo cloudeval_unit_test_passed
# from stackoverflow https://stackoverflow.com/questions/49981601/difference-between-targetport-and-port-in-kubernetes-service-definition
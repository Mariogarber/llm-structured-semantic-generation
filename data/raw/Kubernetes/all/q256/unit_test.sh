kubectl apply -f labeled_code.yaml
sleep 15
kubectl describe service my-service | egrep "http\s+30475/" && kubectl describe service my-service | egrep "metrics\s+31261/" && kubectl describe service my-service | egrep "health\s+30013/" && echo cloudeval_unit_test_passed
# from stackoverflow https://stackoverflow.com/questions/49981601/difference-between-targetport-and-port-in-kubernetes-service-definition
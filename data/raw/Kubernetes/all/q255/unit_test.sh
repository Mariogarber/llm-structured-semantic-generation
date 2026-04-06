kubectl run myapp --image=ealen/echo-server --labels=app.kubernetes.io/name=MyApp
sleep 10
kubectl apply -f labeled_code.yaml
sleep 10
kubectl get svc my-service2 | grep "198.51.100.32" && kubectl describe svc my-service2 | grep "49152/TCP" && echo cloudeval_unit_test_passed
# q1-q5 are from official docs
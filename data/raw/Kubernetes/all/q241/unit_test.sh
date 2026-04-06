kubectl apply -f labeled_code.yaml
sleep 15
kubectl get svc broker-0 | grep "9093:30093/TCP" && echo cloudeval_unit_test_passed
# Stackoverflow https://stackoverflow.com/questions/46456239/how-to-expose-a-headless-kafka-service-for-a-statefulset-externally-in-kubernete
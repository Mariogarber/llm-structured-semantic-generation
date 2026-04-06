kubectl apply -f labeled_code.yaml
sleep 15
kubectl describe ingress minimal-ingress | grep "test-app:5000" && echo cloudeval_unit_test_passed
# Stackoverflow: https://stackoverflow.com/questions/64125048/get-error-unknown-field-servicename-in-io-k8s-api-networking-v1-ingressbacken
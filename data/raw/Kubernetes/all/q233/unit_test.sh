kubectl apply -f labeled_code.yaml
kubectl describe secret secret-basic-auth | grep "password:  10 bytes
username:  5 bytes" && echo cloudeval_unit_test_passed
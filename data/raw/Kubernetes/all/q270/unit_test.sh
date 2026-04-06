kubectl apply -f labeled_code.yaml
kubectl describe statefulset pg-db-storage | grep "Capacity:      1G" && echo cloudeval_unit_test_passed
# https://stackoverflow.com/questions/50804915/kubernetes-size-definitions-whats-the-difference-of-gi-and-g
kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready svc --all --timeout=20s
kubectl describe svc gitlab-metrics | egrep "Port:\s+metrics\s+9252/TCP" && echo cloudeval_unit_test_passed
# from stackoverflow https://stackoverflow.com/questions/52991038/how-to-create-a-servicemonitor-for-prometheus-operator

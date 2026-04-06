kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=priority --timeout=60s
pods=$(kubectl get pods -l name=priority -o=jsonpath='{.items[*].metadata.name}')
kubectl get pod $pods -o=jsonpath='{.spec.priorityClassName}' | grep "system-cluster-critical" && echo cloudeval_unit_test_passed

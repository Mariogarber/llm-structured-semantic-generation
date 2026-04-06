kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l name=readiness --timeout=60s
pods=$(kubectl get pods -l name=readiness -o=jsonpath='{.items[*].metadata.name}')
readiness_path=$(kubectl get pod $pods -o=jsonpath='{.spec.containers[0].readinessProbe.httpGet.path}')
[ "$readiness_path" == "/" ] && echo cloudeval_unit_test_passed

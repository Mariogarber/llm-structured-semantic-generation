kubectl apply -f labeled_code.yaml
kubectl wait deployments --all --for=condition=available --timeout=30s

pods=$(kubectl get pods -l app=test -o=jsonpath='{range .items[*]}{@.metadata.name}{"\n"}{end}')
for pod in $pods; do
    [ "$(kubectl get pod $pod -o=jsonpath='{.spec.containers[0].resources.requests.cpu}')" = "100m" ] && \
    [ "$(kubectl get pod $pod -o=jsonpath='{.spec.containers[0].resources.requests.memory}')" = "100Mi" ] || \
    exit 1
done

kubectl get deployments | grep -q '3/3' && \
echo cloudeval_unit_test_passed

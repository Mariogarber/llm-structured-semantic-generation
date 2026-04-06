kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deployment/sample-dp --timeout=60s 

[ "$(kubectl get deployment sample-dp -o=jsonpath='{.spec.template.metadata.labels.app}')" = "nginx-label" ] && \
[ "$(kubectl get deployment sample-dp -o=jsonpath='{.spec.template.spec.containers[0].image}')" = "nginx:latest" ] && \
echo cloudeval_unit_test_passed

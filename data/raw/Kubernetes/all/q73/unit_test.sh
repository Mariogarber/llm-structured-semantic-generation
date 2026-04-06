kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deploy/custom-deploy --timeout=60s 

[ "$(kubectl get deploy custom-deploy -o=jsonpath='{.spec.replicas}')" -eq 1 ] && \
[ "$(kubectl get deploy custom-deploy -o=jsonpath='{.spec.template.spec.terminationGracePeriodSeconds}')" -eq 10 ] && \
kubectl get endpointslice | grep -q "nginx-service" && \
echo cloudeval_unit_test_passed

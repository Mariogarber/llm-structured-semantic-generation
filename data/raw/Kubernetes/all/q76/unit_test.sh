kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deploy/nginx-deploy --timeout=60s 

[ "$(kubectl get deploy nginx-deploy -o=jsonpath='{.spec.replicas}')" -eq 5 ] && \
[ "$(kubectl get deploy nginx-deploy -o=jsonpath='{.spec.template.spec.containers[0].livenessProbe.httpGet.path}')" = "/" ] && \
[ "$(kubectl get deploy nginx-deploy -o=jsonpath='{.spec.template.spec.containers[0].livenessProbe.httpGet.port}')" -eq 80 ] && \
[ "$(kubectl get deploy nginx-deploy -o=jsonpath='{.spec.template.spec.containers[0].livenessProbe.initialDelaySeconds}')" -eq 5 ] && \
[ "$(kubectl get deploy nginx-deploy -o=jsonpath='{.spec.template.spec.containers[0].livenessProbe.periodSeconds}')" -eq 5 ] && \
echo cloudeval_unit_test_passed

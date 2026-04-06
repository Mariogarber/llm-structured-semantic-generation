kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deploy/mongo-deploy-2 --timeout=60s 
kubectl wait --for=condition=ready pod -l app=db --timeout=60s

[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.spec.replicas}')" -eq 3 ] && \
[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.spec.strategy.type}')" = "RollingUpdate" ] && \
[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.metadata.labels.app}')" = "db" ] && \
[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')" = "100m" ] && \
[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.spec.template.spec.containers[0].resources.requests.memory}')" = "150Mi" ] && \
[ "$(kubectl get deploy mongo-deploy-2 -o=jsonpath='{.spec.template.spec.containers[0].resources.limits.memory}')" = "200Mi" ] && \
echo cloudeval_unit_test_passed

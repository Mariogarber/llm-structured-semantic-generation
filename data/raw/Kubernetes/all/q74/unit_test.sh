kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=available deploy/mongo-deploy --timeout=60s 
kubectl wait --for=condition=ready pod -l app=db --timeout=60s

[ "$(kubectl get deployment mongo-deploy -o=jsonpath='{.spec.replicas}')" -eq 5 ] 
[ "$(kubectl get deployment mongo-deploy -o=jsonpath='{.spec.strategy.type}')" = "RollingUpdate" ] || exit 1
[ "$(kubectl get deployment mongo-deploy -o=jsonpath='{.metadata.labels.app}')" = "db" ] || exit 1
echo cloudeval_unit_test_passed

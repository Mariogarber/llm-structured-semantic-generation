kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l tier=backend --timeout=60s

[ "$(kubectl get rs backend -o=jsonpath='{.spec.replicas}')" -eq 2 ] && \
[ "$(kubectl get rs backend -o=jsonpath='{.spec.selector.matchLabels.tier}')" = "backend" ] && \
[ "$(kubectl get rs backend -o=jsonpath='{.metadata.labels.app}')" = "database" ] && \
[ "$(kubectl get rs backend -o=jsonpath='{.metadata.labels.tier}')" = "backend" ] && \
[ "$(kubectl get rs backend -o=jsonpath='{.spec.template.spec.containers[0].image}')" = "mongo:latest" ] && \
[ "$(kubectl get rs backend -o=jsonpath='{.spec.template.spec.containers[0].name}')" = "db-container" ] && \
echo cloudeval_unit_test_passed

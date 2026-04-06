kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l tier=old-backend --timeout=60s
kubectl wait --for=condition=Ready pod -l tier=backend --timeout=60s

POD1_LABEL=$(kubectl get pod pod1 -o=jsonpath='{.metadata.labels.tier}')
POD2_LABEL=$(kubectl get pod pod2 -o=jsonpath='{.metadata.labels.tier}')
RS_REPLICAS=$(kubectl get rs backend -o=jsonpath='{.spec.replicas}')
RS_LABEL=$(kubectl get rs backend -o=jsonpath='{.metadata.labels.tier}')
RS_SELECTOR=$(kubectl get rs backend -o=jsonpath='{.spec.selector.matchLabels.tier}')
RS_PODS=$(kubectl get pods --selector=tier=backend --no-headers=true | wc -l)

[ "$POD1_LABEL" = "old-backend" ] && \
[ "$POD2_LABEL" = "old-backend" ] && \
[ "$RS_REPLICAS" -eq 3 ] && \
[ "$RS_LABEL" = "backend" ] && \
[ "$RS_SELECTOR" = "backend" ] && \
[ "$RS_PODS" -eq 3 ] && \
echo cloudeval_unit_test_passed

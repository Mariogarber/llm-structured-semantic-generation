kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod -l app=my-app --timeout=60s
RS_REPLICAS=$(kubectl get rs myrs -o=jsonpath='{.spec.replicas}')
RS_SELECTOR_APP=$(kubectl get rs myrs -o=jsonpath='{.spec.selector.matchLabels.app}')
RS_SELECTOR_TIER=$(kubectl get rs myrs -o=jsonpath='{.spec.selector.matchLabels.tier}')
RS_SELECTOR_ENV=$(kubectl get rs myrs -o=jsonpath='{.spec.selector.matchExpressions[0].key}')
POD_IMAGE=$(kubectl get rs myrs -o=jsonpath='{.spec.template.spec.containers[0].image}')
POD_PORT=$(kubectl get rs myrs -o=jsonpath='{.spec.template.spec.containers[0].ports[0].containerPort}')

[ "$RS_REPLICAS" -eq 3 ] && \
[ "$RS_SELECTOR_APP" = "my-app" ] && \
[ "$RS_SELECTOR_TIER" = "backend" ] && \
[ "$RS_SELECTOR_ENV" = "env" ] && \
[ "$POD_IMAGE" = "nginx:latest" ] && \
[ "$POD_PORT" -eq 80 ] && \
echo cloudeval_unit_test_passed

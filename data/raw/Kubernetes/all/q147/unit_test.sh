kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/readonly-pod --timeout=60s
VOLUME=$(kubectl get pod readonly-pod -o=jsonpath='{.spec.volumes[0].configMap.name}')
MSG=$(kubectl exec -it readonly-pod -- /bin/sh -c "cat /etc/config/my-config")

[ "$VOLUME" = "my-config" ] && \
[ "$MSG" = "It is read-only." ] && \
echo cloudeval_unit_test_passed

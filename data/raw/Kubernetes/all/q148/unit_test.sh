kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pods/multiconfig-pod --timeout=60s
VOLUME1=$(kubectl get pod multiconfig-pod -o=jsonpath='{.spec.volumes[0].configMap.name}')
VOLUME2=$(kubectl get pod multiconfig-pod -o=jsonpath='{.spec.volumes[1].configMap.name}')

[ "$VOLUME1" = "config1" ] && \
[ "$VOLUME2" = "config2" ] && \
[ "$(kubectl exec multiconfig-pod -- cat /etc/config1/key1)" = 'file1value' ] && \
[ "$(kubectl exec multiconfig-pod -- cat /etc/config2/key2)" = 'file2value' ] && \
echo cloudeval_unit_test_passed

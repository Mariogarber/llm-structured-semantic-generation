kubectl apply -f labeled_code.yaml

[ "$(kubectl get pvc wp-pv-claim -o jsonpath='{.spec.accessModes[0]}')" = "ReadWriteOnce" ] && \
[ "$(kubectl get pvc wp-pv-claim -o jsonpath='{.spec.resources.requests.storage}')" = "2Gi" ] && \
echo cloudeval_unit_test_passed

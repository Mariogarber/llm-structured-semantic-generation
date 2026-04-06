kubectl apply -f labeled_code.yaml
sleep 5
kubectl wait --for=condition=initialized pod -l app=zk --timeout=60s

[ "$(kubectl get sts zk -o jsonpath='{.spec.updateStrategy.type}')" = "RollingUpdate" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.podManagementPolicy}')" = "OrderedReady" ] && \
[ "$(kubectl get poddisruptionbudget zk-pdb -o jsonpath='{.spec.maxUnavailable}')" = "1" ] && \
echo cloudeval_unit_test_passed

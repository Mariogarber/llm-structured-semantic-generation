kubectl apply -f labeled_code.yaml

[ "$(kubectl get sc local-sc -o jsonpath='{.provisioner}')" = "kubernetes.io/no-provisioner" ] && \
[ "$(kubectl get sc local-sc -o jsonpath='{.volumeBindingMode}')" = "WaitForFirstConsumer" ] && \
echo cloudeval_unit_test_passed

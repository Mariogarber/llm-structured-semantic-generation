kubectl apply -f labeled_code.yaml

[ "$(kubectl get sc dynamic-sc -o jsonpath='{.provisioner}')" = "kubernetes.io/aws-ebs" ] && \
[ "$(kubectl get sc dynamic-sc -o jsonpath='{.parameters.type}')" = "gp2" ] && \
[ "$(kubectl get sc dynamic-sc -o jsonpath='{.reclaimPolicy}')" = "Retain" ] && \
[ "$(kubectl get sc dynamic-sc -o jsonpath='{.volumeBindingMode}')" = "Immediate" ] && \
echo cloudeval_unit_test_passed

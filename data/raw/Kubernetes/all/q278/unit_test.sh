kubectl apply -f labeled_code.yaml

[ "$(kubectl get sc test-sc -o jsonpath='{.provisioner}')" = "kubernetes.io/aws-ebs" ] && \
[ "$(kubectl get sc test-sc -o jsonpath='{.parameters.type}')" = "gp2" ] && \
[ "$(kubectl get sc test-sc -o jsonpath='{.reclaimPolicy}')" = "Retain" ] && \
[ "$(kubectl get sc test-sc -o jsonpath='{.allowVolumeExpansion}')" = "true" ] && \
[ "$(kubectl get sc test-sc -o jsonpath='{.mountOptions[0]}')" = "debug" ] && \
[ "$(kubectl get sc test-sc -o jsonpath='{.volumeBindingMode}')" = "Immediate" ] && \
echo cloudeval_unit_test_passed

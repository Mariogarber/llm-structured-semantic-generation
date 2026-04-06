kubectl apply -f labeled_code.yaml

[ "$(kubectl get sc zone-sc -o jsonpath='{.provisioner}')" = "kubernetes.io/gce-pd" ] && \
[ "$(kubectl get sc zone-sc -o jsonpath='{.volumeBindingMode}')" = "WaitForFirstConsumer" ] && \
[ "$(kubectl get sc zone-sc -o jsonpath='{.allowedTopologies[0].matchLabelExpressions[0].key}')" = "topology.kubernetes.io/zone" ] && \
[ "$(kubectl get sc zone-sc -o jsonpath='{.allowedTopologies[0].matchLabelExpressions[0].values[0]}')" = "us-central-1a" ] && \
[ "$(kubectl get sc zone-sc -o jsonpath='{.allowedTopologies[0].matchLabelExpressions[0].values[1]}')" = "us-central-1b" ] && \
echo cloudeval_unit_test_passed

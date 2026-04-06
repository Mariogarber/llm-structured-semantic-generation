kubectl apply -f labeled_code.yaml
sleep 30

[ "$(kubectl get sc local -o jsonpath='{.provisioner}')" = "k8s.io/minikube-hostpath" ] && \
[ "$(kubectl get pvc claim1 -o jsonpath='{.spec.resources.requests.storage}')" = "3Gi" ] && \
[ "$(kubectl get pvc claim1 -o jsonpath='{.status.phase}')" = "Bound" ] && \
echo cloudeval_unit_test_passed

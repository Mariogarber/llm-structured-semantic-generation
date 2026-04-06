kubectl apply -f labeled_code.yaml

[ "$(kubectl get sc fast-storage -o jsonpath='{.provisioner}')" = "standard" ] && \
[ "$(kubectl get pvc app-pvc -o jsonpath='{.spec.accessModes[0]}')" = "ReadWriteOnce" ] && \
[ "$(kubectl get pvc app-pvc -o jsonpath='{.spec.storageClassName}')" = "fast-storage" ] && \
[ "$(kubectl get pvc app-pvc -o jsonpath='{.spec.resources.requests.storage}')" = "100Mi" ] && \
echo cloudeval_unit_test_passed

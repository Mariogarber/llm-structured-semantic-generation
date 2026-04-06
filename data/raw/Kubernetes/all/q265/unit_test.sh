kubectl apply -f labeled_code.yaml
sleep 10
kubectl wait --for=condition=ready pod -l app=my-cassandra --timeout=60s

[ "$(kubectl get sc fast -o jsonpath='{.provisioner}')" = "k8s.io/minikube-hostpath" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.volumeClaimTemplates[0].metadata.name}')" = "cassandra-data" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.volumeClaimTemplates[0].spec.resources.requests.storage}')" = "100Mi" ] && \
[ "$(kubectl get sts my-cassandra -o jsonpath='{.spec.volumeClaimTemplates[0].spec.accessModes[0]}')" = "ReadWriteOnce" ] && \
echo cloudeval_unit_test_passed

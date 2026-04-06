kubectl apply -f labeled_code.yaml

kubectl get svc | grep -q "etcd-discovery" && \
[ "$(kubectl get svc etcd-discovery -o=jsonpath='{.spec.type}')" = "ClusterIP" ] && \
[ "$(kubectl get svc etcd-discovery -o=jsonpath='{.spec.ports[0].port}')" = "2379" ] && \
[ "$(kubectl get svc etcd-discovery -o=jsonpath='{.metadata.labels.name}')" = "etcd" ] && \
[ "$(kubectl get svc etcd-discovery -o=jsonpath='{.spec.sessionAffinity}')" = "None" ] && \
echo cloudeval_unit_test_passed

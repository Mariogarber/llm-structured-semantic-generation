kubectl apply -f labeled_code.yaml
sleep 5
kubectl wait --for=condition=initialized pod -l app=zk --timeout=60s

[ "$(kubectl get sts zk -o jsonpath='{.spec.replicas}')" = "3" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.template.spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey}')" = "kubernetes.io/hostname" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.template.spec.containers[0].livenessProbe.exec.command[2]}')" = "zookeeper-ready 2181" ] && \
[ "$(kubectl get sts zk -o jsonpath='{.spec.template.spec.containers[0].readinessProbe.exec.command[2]}')" = "zookeeper-ready 2181" ] && \
echo cloudeval_unit_test_passed

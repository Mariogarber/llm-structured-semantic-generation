kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=Ready pod/sim-pod --timeout=60s

[ "$(kubectl get pod sim-pod -o=jsonpath='{.spec.containers[0].volumeMounts[0].mountPath}')" = "/mnt/local" ] && \
[ "$(kubectl get pod sim-pod -o=jsonpath='{.spec.volumes[0].hostPath.path}')" = "/tmp/local" ] && \
[ "$(kubectl get secret sim-secret -o jsonpath='{.metadata.name}')" = "sim-secret" ] && \
[ "$(kubectl get secret sim-secret -o=jsonpath='{.data.username}' | base64 -d)" = "simulated" ] && \
[ "$(kubectl get secret sim-secret -o=jsonpath='{.data.password}' | base64 -d)" = "secret" ] && \
echo cloudeval_unit_test_passed

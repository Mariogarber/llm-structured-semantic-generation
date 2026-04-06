kubectl apply -f labeled_code.yaml
kubectl wait --for=condition=ready pod -l app=agent-test --timeout=60s
sleep 10

pods=$(kubectl get pods -l app=agent-test -o name)
for pod in $pods; do
  [ $(kubectl get $pod -o=jsonpath='{.spec.hostNetwork}') = "true" ] && \
  [ $(kubectl get $pod -o=jsonpath='{.spec.hostPID}') = "true" ] && \
  [ $(kubectl get $pod -o=jsonpath='{.spec.containers[0].securityContext.privileged}') = "true" ] && \
  kubectl get $pod -o=jsonpath="{.spec.containers[0].volumeMounts[?(@.name=='dev-vol')].mountPath}" | grep -q "/host/dev" && \
  kubectl get $pod -o=jsonpath="{.spec.containers[0].volumeMounts[?(@.name=='proc-vol')].mountPath}" | grep -q "/host/proc" && \
  kubectl get $pod -o=jsonpath="{.spec.containers[0].volumeMounts[?(@.name=='usr-vol')].mountPath}" | grep -q "/host/usr" ||
  exit 1
done

[ "$(kubectl get rc test-2 --output=jsonpath='{.status.readyReplicas}')" = 5 ] && \
echo cloudeval_unit_test_passed
